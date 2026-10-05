# The Interlocutor Effect

Companion experiments for **The Interlocutor Effect: Why LLMs Leak More Personal Data to Agents Than Humans**, by Faouzi El Yagoubi, Godwin Badu-Marfo, and Ranwa Al Mallah.

Published in the **2026 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW)** (IWPE 2026).

- [Conference paper (DOI)](https://doi.org/10.1109/EuroSPW72509.2026.00008)
- [Institutional publication record](https://publications.polymtl.ca/80261/)
- [Public manuscript, arXiv v1](https://arxiv.org/html/2606.09844v1)

This README documents the conference study. The linked arXiv v1 is the public manuscript used to check the results below; it is not identified here as a separate extended study.

## Cite the paper

Use the conference DOI when citing the published study. Machine-readable metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{elyagoubi2026interlocutor,
  author    = {El Yagoubi, Faouzi and Badu-Marfo, Godwin and Al Mallah, Ranwa},
  title     = {The Interlocutor Effect: Why {LLMs} Leak More Personal Data to Agents Than Humans},
  booktitle = {2026 IEEE European Symposium on Security and Privacy Workshops (EuroS\&PW)},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/EuroSPW72509.2026.00008},
  url       = {https://doi.org/10.1109/EuroSPW72509.2026.00008}
}
```

## Study and published results

The experiment crosses **recipient framing (human/agent)** with **output format (text/JSON)**, using synthetic PII and the AgentLeak framework. It includes 222 scenarios and 3,464 responses across four target models; Mistral uses 200 scenarios. Qwen-2.5-7B serves as an evaluator, not a fifth target model.

Leakage rates (%) from Table III of the [public manuscript](https://arxiv.org/html/2606.09844v1#S4.T3):

| Target model | Human text (`C_HT`) | Agent text (`C_AT`) | Human JSON (`C_HJ`) | Agent JSON (`C_AJ`) |
|---|---:|---:|---:|---:|
| GPT-4o | 82.9 | 95.5 | 91.4 | 90.5 |
| Claude 3.5 Sonnet | 89.2 | 96.4 | 80.2 | 90.5 |
| Llama 3.3 70B | 68.0 | 91.0 | 69.4 | 59.9 |
| Mistral Large | 94.0 | 96.5 | 90.0 | 92.5 |

The paper reports an aggregate text-condition increase of **11.5 percentage points**, with heterogeneous model effects. These are adversarial benchmark rates, not estimates of ordinary deployment leakage.

In the GPT-4o ablation (100 scenarios), agent versus technical-human framing is **not statistically significant** (`p = 0.259`). Technical context contributes to the effect; agent identity is not isolated as its sole cause. The Llama ablation does not confirm the effect. The proposed attention mechanism remains preliminary, not an established general explanation.

## What is in this repository

Inventory checked against the committed JSON files; published results and smoke artifacts must be kept separate.

| File(s) | Contents and relationship to the paper |
|---|---|
| [results/benchmark.json](results/benchmark.json) | 3,476 records: **3,464 conference-model records plus 12 GPT-4o-mini records**. The conference subset covers 222 unique scenario IDs and reproduces all per-model rates above. Filter to the four named target models before aggregating. Records include responses and detected fields; this is not a standalone scenario-generator input export. |
| [results/ablation_gpt4o.json](results/ablation_gpt4o.json) | 300 records, 100 scenarios × three contexts. Mean leaked-field counts round to the paper's 4.2 (human), 4.8 (technical human), and 5.0 (agent). |
| [results/ablation_v2.json](results/ablation_v2.json) | 90 Llama 3.3 70B records, 30 scenarios × three contexts. |
| [results/interlocutor_results.json](results/interlocutor_results.json), [results/checkpoint.json](results/checkpoint.json) | 20 GPT-4o-mini records covering five scenarios and four contexts: smoke artifacts. |
| [results/factorial_table.json](results/factorial_table.json), [results/model_table.json](results/model_table.json), [results/vertical_table.json](results/vertical_table.json), [results/stats.json](results/stats.json) | Smoke summaries (`n = 5` per condition), **not the published tables or statistics**. |
| [results/traces/](results/traces/) | 20 GPT-4o-mini smoke traces, not traces for all conference-model responses. |
| [results/smoke.json](results/smoke.json) | Separate partial smoke output: 14 GPT-4o-mini records across four scenario IDs. |
| `results/activation_patching_results_run{2,3,4}.json` | Exploratory Llama-3.1-8B-Instruct runs. All three record `hypothesis_supported: false`; these files should not be presented as definitive validation of the proposed mechanism. |

The scenario coverage in the conference subset is healthcare (50), finance (58), legal (58), and corporate (56). No claim is made that every published statistic or mechanistic conclusion has been independently reproduced from these artifacts.

## Scripts and execution status

- [benchmark.py](benchmark.py): recipient × format benchmark and saved-result analysis.
- [ablation_run.py](ablation_run.py): human, technical-human, and agent comparison.
- [audit_benchmark.py](audit_benchmark.py): audits the entire `results/benchmark.json`, including its extra GPT-4o-mini records; its aggregate output is not automatically the conference subset.
- [attention_probe.py](attention_probe.py), [activation_patching.py](activation_patching.py), [vertex_activation_patching.py](vertex_activation_patching.py): exploratory mechanistic experiments.
- [launch_vertex_job.sh](launch_vertex_job.sh), [watch_vertex_job.sh](watch_vertex_job.sh): Vertex AI job helpers; inspect their configuration before use.

The benchmark imports [AgentLeak](https://github.com/Privatris/AgentLeak) and prepends `ROOT.parent.parent / "AgentLeak"` to the Python import path. A compatible AgentLeak installation or checkout is required, including its generator and detection modules. [vertex_requirements.txt](vertex_requirements.txt) contains the Vertex experiment dependencies; it is **not a complete benchmark environment specification**. No end-to-end installation recipe is verified here.

The following flags exist in the checked-in CLI. These are usage examples, not verified end-to-end reproductions. Live runs require OpenRouter credentials (`OPENROUTER_API_KEY`), model access, and the detector dependencies/configuration. They make external API calls and may incur costs.

```bash
# Live smoke run: five requested scenarios, GPT-4o-mini; writes results/smoke.json
python benchmark.py --smoke

# Live small benchmark; choose a separate output to preserve committed results
python benchmark.py --num_scenarios 10 --models openai/gpt-4o --out local_benchmark.json

# Analyze a saved file; imports still require AgentLeak
python benchmark.py --analyze results/benchmark.json

# Live ablation; choose a separate output
python ablation_run.py --num_scenarios 30 --model openai/gpt-4o --out local_ablation.json
```

The saved ablation files use `scenario_id`/`condition`, while the current `ablation_run.py --analyze` implementation expects `id`/`cond`. Direct reanalysis of those files therefore requires an explicit schema conversion; the command is not a ready-made reproduction of Table IV. Benchmark analysis also aggregates every model in its input, so filter the additional GPT-4o-mini records for conference comparisons.
