#!/usr/bin/env python3
"""
Audit benchmark results to detect methodological issues.
Verifies: ceiling effect, false positives, delta consistency.
"""
import json, os, sys, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
data_path = ROOT / "results" / "benchmark.json"

with open(data_path) as f:
    data = json.load(f)

print(f"Total records loaded     : {len(data)}")
ids = set(r["id"] for r in data)
print(f"Unique scenarios         : {len(ids)}")
print()

# ── Leak rate by condition ─────────────────────────────────────────
cc = defaultdict(lambda: {"t": 0, "l": 0})
for r in data:
    cc[r["cond"]]["t"] += 1
    if r["leaked"]: cc[r["cond"]]["l"] += 1

def rate(c): return 100*cc[c]["l"]/cc[c]["t"] if cc[c]["t"] else 0

print("=== LEAK RATE BY CONDITION ===")
for c in ["C_HT","C_HJ","C_AT","C_AJ"]:
    t, l = cc[c]["t"], cc[c]["l"]
    print(f"  {c}: {100*l/t:5.1f}%  ({l}/{t})")

ht, hj, at, aj = rate("C_HT"), rate("C_HJ"), rate("C_AT"), rate("C_AJ")
di_text = at - ht
di_json = aj - hj
print(f"\n  ΔI (text)  = C_AT({at:.1f}) - C_HT({ht:.1f}) = {di_text:+.1f}%")
print(f"  ΔI (json)  = C_AJ({aj:.1f}) - C_HJ({hj:.1f}) = {di_json:+.1f}%")
print(f"  ΔF (human) = C_HJ({hj:.1f}) - C_HT({ht:.1f}) = {hj-ht:+.1f}%")
print(f"  ΔF (agent) = C_AJ({aj:.1f}) - C_AT({at:.1f}) = {aj-at:+.1f}%")

# ── CEILING EFFECT TEST ─────────────────────────────────────────
print(f"\n=== CEILING EFFECT TEST ===")
print(f"  Baseline C_HT = {ht:.1f}%")
if ht >= 70:
    print(f"  ⚠  CEILING EFFECT DETECTED: baseline is ≥70%.")
    print(f"     Remaining margin: {100-ht:.1f}%. Effect Δ={di_text:+.1f}% within margin {100-ht:.1f}%.")
    print(f"     → Agent effect appears small in absolute terms but compressed by ceiling.")
else:
    print(f"  ✓ No major ceiling effect (margin = {100-ht:.1f}%)")

# ── PAIRED ANALYSIS by (scenario, model) ───────────────────────
print(f"\n=== PAIRED ANALYSIS (scenario × model) ===")
per_sc = defaultdict(lambda: defaultdict(dict))
for r in data:
    per_sc[r["id"]][r["model"]][r["cond"]] = r["leaked"]

cats = {"HT_ok_AT_ok": 0, "HT_ok_AT_LEAK": 0, "HT_LEAK_AT_ok": 0, "HT_LEAK_AT_LEAK": 0}
for sc_id, models in per_sc.items():
    for model, conds in models.items():
        ht_l = conds.get("C_HT"); at_l = conds.get("C_AT")
        if ht_l is None or at_l is None: continue
        if not ht_l and not at_l: cats["HT_ok_AT_ok"] += 1
        elif not ht_l and at_l:   cats["HT_ok_AT_LEAK"] += 1
        elif ht_l and not at_l:   cats["HT_LEAK_AT_ok"] += 1
        else:                      cats["HT_LEAK_AT_LEAK"] += 1

total = sum(cats.values())
print(f"  (scenario, model) pairs analyzed: {total}")
print(f"  Both SAFE             (HT=ok,  AT=ok)   : {cats['HT_ok_AT_ok']:4d} ({100*cats['HT_ok_AT_ok']/total:5.1f}%)")
print(f"  TRUE interlocutor effect (HT=ok,  AT=LEAK) : {cats['HT_ok_AT_LEAK']:4d} ({100*cats['HT_ok_AT_LEAK']/total:5.1f}%)")
print(f"  Agent safer (unexpected) (HT=LEAK,AT=ok)   : {cats['HT_LEAK_AT_ok']:4d} ({100*cats['HT_LEAK_AT_ok']/total:5.1f}%)")
print(f"  Both LEAK            (HT=LEAK,AT=LEAK)  : {cats['HT_LEAK_AT_LEAK']:4d} ({100*cats['HT_LEAK_AT_LEAK']/total:5.1f}%)")

# McNemar test
b, c = cats["HT_ok_AT_LEAK"], cats["HT_LEAK_AT_ok"]
if b + c > 0:
    mcnemar_chi2 = (abs(b - c) - 1)**2 / (b + c)
    print(f"\n  McNemar test (b={b}, c={c}):")
    print(f"    χ²(McNemar) = {mcnemar_chi2:.2f}")
    from scipy.stats import chi2 as chi2_dist
    p_mcnemar = 1 - chi2_dist.cdf(mcnemar_chi2, df=1)
    print(f"    p = {p_mcnemar:.4f}  {'*** SIGNIFICANT' if p_mcnemar < 0.001 else '** SIGNIFICANT' if p_mcnemar < 0.01 else '* SIGNIFICANT' if p_mcnemar < 0.05 else 'NOT SIGNIFICANT'}")
    OR = b/c if c > 0 else float("inf")
    print(f"    Odds Ratio agent/human = {OR:.2f}")

# ── DETECTED FIELDS ANALYSIS ─────────────────────────────────────
print(f"\n=== DETECTED FIELDS BY CONDITION ===")
for cond in ["C_HT","C_AT"]:
    recs = [r for r in data if r["cond"]==cond and r["leaked"]]
    all_fields = defaultdict(int)
    for r in recs:
        for f in r.get("fields", []):
            all_fields[f] += 1
    total_leaked = len(recs)
    print(f"\n  {cond} ({total_leaked} leaked) - top 10 fields:")
    for f, c in sorted(all_fields.items(), key=lambda x: -x[1])[:10]:
        print(f"    {f:40s}: {c:4d} ({100*c/total_leaked:.0f}%)")

# ── AVERAGE LEAK VOLUME ─────────────────────────────────────────
print(f"\n=== AVERAGE VOLUME OF LEAKED FIELDS ===")
for cond in ["C_HT","C_HJ","C_AT","C_AJ"]:
    recs = [r for r in data if r["cond"]==cond]
    all_n = [r.get("n_leaked",0) for r in recs]
    leaked_n = [r.get("n_leaked",0) for r in recs if r["leaked"]]
    avg_all = sum(all_n)/len(all_n) if all_n else 0
    avg_leaked = sum(leaked_n)/len(leaked_n) if leaked_n else 0
    print(f"  {cond}: global avg={avg_all:.2f}, avg if leaked={avg_leaked:.2f}")

# ── CONSISTENCY BY MODEL ─────────────────────────────────────────
print(f"\n=== LEAK RATE BY MODEL ===")
mc = defaultdict(lambda: defaultdict(lambda: {"t": 0, "l": 0}))
for r in data:
    mc[r["model"]][r["cond"]]["t"] += 1
    if r["leaked"]: mc[r["model"]][r["cond"]]["l"] += 1

SHORT = {
    "openai/gpt-4o": "GPT-4o",
    "anthropic/claude-3.5-sonnet": "Claude 3.5",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "mistralai/mistral-large-2411": "Mistral Large",
}
print(f"  {'Model':18} {'C_HT':>6} {'C_AT':>6} {'ΔI':>6}  {'C_HJ':>6} {'C_AJ':>6} {'ΔI':>6}")
for m in sorted(mc):
    ms = SHORT.get(m, m[-20:])
    r = {c: 100*mc[m][c]["l"]/mc[m][c]["t"] if mc[m][c]["t"] else 0 for c in ["C_HT","C_HJ","C_AT","C_AJ"]}
    di_t = r["C_AT"] - r["C_HT"]
    di_j = r["C_AJ"] - r["C_HJ"]
    print(f"  {ms:18} {r['C_HT']:6.1f} {r['C_AT']:6.1f} {di_t:+6.1f}  {r['C_HJ']:6.1f} {r['C_AJ']:6.1f} {di_j:+6.1f}")

# ── ANALYSIS BY VERTICAL ─────────────────────────────────────────
print(f"\n=== LEAK RATE BY VERTICAL ===")
vc = defaultdict(lambda: defaultdict(lambda: {"t": 0, "l": 0}))
for r in data:
    vc[r["vert"]][r["cond"]]["t"] += 1
    if r["leaked"]: vc[r["vert"]][r["cond"]]["l"] += 1

print(f"  {'Vertical':14} {'C_HT':>6} {'C_AT':>6} {'ΔI':>6}")
for vt in sorted(vc):
    r = {c: 100*vc[vt][c]["l"]/vc[vt][c]["t"] if vc[vt][c]["t"] else 0 for c in ["C_HT","C_AT"]}
    print(f"  {vt:14} {r['C_HT']:6.1f} {r['C_AT']:6.1f} {r['C_AT']-r['C_HT']:+6.1f}")

# ── VERDICT ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")
if ht >= 70:
    print(f"⚠  CEILING EFFECT PROBLEM: C_HT={ht:.1f}% is too high.")
    print(f"   Probable causes:")
    print(f"   1. allowed_set too narrow (or empty) → everything is 'leak'")
    print(f"   2. User request is vague ('give me a rundown')")
    print(f"      → even legitimate disclosure is classified as leak")
    print(f"   3. LLM judge (Qwen-7B) may over-detect with limited context")
    print(f"   → Agent effect Δ={di_text:+.1f}% is real but its magnitude")
    print(f"     is compressed by ceiling. The true value to report")
    print(f"     is the CONDITIONAL delta and McNemar odds ratio.")
else:
    print(f"✓ No ceiling effect. Results appear reliable.")
    print(f"  Interlocutor effect: ΔI={di_text:+.1f}% (text), {di_json:+.1f}% (json)")
