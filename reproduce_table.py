#!/usr/bin/env python3
"""Recompute conference Table III offline using only Python's standard library."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

DATA = Path(__file__).resolve().parent / 'results' / 'benchmark.json'
SHA256 = '308e1d7b967194790d0f3d6dcd5a404cf66d3456f9601d4f7ddc3ba89a5a9727'
CONDITIONS = ('C_HT', 'C_AT', 'C_HJ', 'C_AJ')
MODELS = {
    'openai/gpt-4o': ('GPT-4o', 222, (82.9, 95.5, 91.4, 90.5)),
    'anthropic/claude-3.5-sonnet': ('Claude 3.5 Sonnet', 222, (89.2, 96.4, 80.2, 90.5)),
    'meta-llama/llama-3.3-70b-instruct': ('Llama 3.3 70B', 222, (68.0, 91.0, 69.4, 59.9)),
    'mistralai/mistral-large-2411': ('Mistral Large', 200, (94.0, 96.5, 90.0, 92.5)),
}
SMOKE = 'openai/gpt-4o-mini'
DOMAINS = {'healthcare': 50, 'finance': 58, 'legal': 58, 'corporate': 56}


def summarize(records):
    """Validate the saved study structure and compute rates, never infer labels."""
    if not isinstance(records, list):
        raise ValueError('Expected a JSON list of benchmark records')
    seen, scenarios, excluded = set(), {}, Counter()
    groups = {(m, c): [] for m in MODELS for c in CONDITIONS}
    ids = {(m, c): set() for m in MODELS for c in CONDITIONS}
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f'Record {index}: expected an object')
        for key in ('id', 'model', 'cond', 'vert'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f'Record {index}: missing/invalid {key}')
        if type(row.get('leaked')) is not bool:
            raise ValueError(f'Record {index}: leaked must be a JSON boolean')
        model, condition, sid = row['model'], row['cond'], row['id']
        if model not in MODELS and model != SMOKE:
            raise ValueError(f'Record {index}: unexpected model')
        if condition not in CONDITIONS or row['vert'] not in DOMAINS:
            raise ValueError(f'Record {index}: unexpected condition/domain')
        key = (model, condition, sid)
        if key in seen:
            raise ValueError(f'Record {index}: duplicate model/condition/scenario')
        seen.add(key)
        if model == SMOKE:
            excluded[model] += 1
            continue
        if sid in scenarios and scenarios[sid] != row['vert']:
            raise ValueError('Inconsistent domain for a scenario')
        scenarios[sid] = row['vert']
        groups[model, condition].append(row['leaked'])
        ids[model, condition].add(sid)
    if len(scenarios) != 222 or dict(Counter(scenarios.values())) != DOMAINS:
        raise ValueError('Conference scenario/domain coverage differs from the study')
    result = []
    for model, (label, expected_n, published) in MODELS.items():
        cells = []
        for condition in CONDITIONS:
            values = groups[model, condition]
            if len(values) != expected_n or ids[model, condition] != ids[model, 'C_HT']:
                raise ValueError(f'{model}: incomplete or mismatched condition coverage')
            cells.append({'condition': condition, 'n': len(values),
                          'leaked': sum(values), 'rate': round(100 * sum(values) / len(values), 1)})
        result.append({'model': label, 'cells': cells,
                       'matches_published': tuple(c['rate'] for c in cells) == published})
    return {'conference_records': sum(len(v) for v in groups.values()),
            'scenarios': len(scenarios), 'excluded_records': sum(excluded.values()),
            'excluded_models': dict(excluded), 'table': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA, help='saved benchmark JSON')
    parser.add_argument('--check', action='store_true', help='require exact archived checksum and published rates')
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if args.check and digest != SHA256:
            raise ValueError('SHA-256 differs from the archived benchmark.json')
        report = summarize(json.loads(raw))
        if args.check and not all(r['matches_published'] for r in report['table']):
            raise ValueError('Computed rates differ from published Table III')
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    print(f'SHA-256: {digest}')
    print(f"Conference: {report['conference_records']} records, {report['scenarios']} scenarios")
    print(f"Excluded non-conference records: {report['excluded_records']} ({SMOKE})")
    print('\n| Model | n per condition | Human text C_HT | Agent text C_AT | Human JSON C_HJ | Agent JSON C_AJ |')
    print('|---|---:|---:|---:|---:|---:|')
    for row in report['table']:
        rates = ' | '.join(f"{cell['rate']:.1f}" for cell in row['cells'])
        print(f"| {row['model']} | {row['cells'][0]['n']} | {rates} |")
    print('\nRates are percentages derived from saved boolean leakage labels; no model or detector is run.')
    print('Published Table III match: ' + ('yes' if all(r['matches_published'] for r in report['table']) else 'NO'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
