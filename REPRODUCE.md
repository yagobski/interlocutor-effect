# Reproduce the conference table offline

From the repository root, run:

```bash
python3 reproduce_table.py --check
```

Requires Python 3.9 or newer and its standard library only. No installation, credentials, AgentLeak checkout, model access, or network request is needed. The script reads `results/benchmark.json`, prints Markdown to stdout, and never modifies the source data. It also works from another directory when invoked using the script's absolute path.

The output reproduces the four model rows of Table III in the [conference study's public manuscript](https://arxiv.org/html/2606.09844v1#S4.T3):

| Model | n per condition | Human text C_HT | Agent text C_AT | Human JSON C_HJ | Agent JSON C_AJ |
|---|---:|---:|---:|---:|---:|
| GPT-4o | 222 | 82.9 | 95.5 | 91.4 | 90.5 |
| Claude 3.5 Sonnet | 222 | 89.2 | 96.4 | 80.2 | 90.5 |
| Llama 3.3 70B | 222 | 68.0 | 91.0 | 69.4 | 59.9 |
| Mistral Large | 200 | 94.0 | 96.5 | 90.0 | 92.5 |

Values are percentages, computed as 100 × saved `leaked: true` records / records in each model-condition cell, rounded to one decimal. There are 3,464 included records over 222 scenario IDs; 12 GPT-4o-mini records are explicitly excluded. Columns follow the paper's order, not the input record order. This command does not compute an unweighted average across models or reproduce inferential statistics.

Every run checks required field types, boolean leakage labels, recognized models/conditions/domains, duplicate scenario-model-condition keys, matched scenario sets across conditions, expected per-model cell counts, and overall scenario/domain coverage. Unknown models fail validation instead of being silently discarded. Only GPT-4o-mini is excluded.

`--check` additionally requires the exact SHA-256 of the committed archive and agreement with all 16 published per-model rates. It exits nonzero on failure. The digest is a reproducibility fingerprint, not proof of authenticity or scientific correctness.

To inspect a copy with the same study structure but different bytes, use `python3 reproduce_table.py --input /path/to/benchmark.json` without `--check`. Structural validation still applies, and output explicitly reports whether rates match the paper. This is not a generic analyzer for arbitrary experiments.

Run deterministic tests with:

```bash
python3 -m unittest -v test_reproduce_table
```

The tests cover published counts/rates, shuffled input, exclusion of smoke labels, malformed/duplicate/missing records, checksum failure, invocation outside the repository, and unchanged input bytes.

This is an arithmetic reproduction of saved labels. It does not rerun prompts, regenerate scenarios, validate leakage detection, establish a causal mechanism, or reproduce the ablation's significance tests. The agent-versus-technical-human contrast remains non-significant in the paper. No new data or scientific claims are introduced.
