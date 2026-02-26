# The Interlocutor Effect: Experiments

Reproducible experiments for the paper:

**The Interlocutor Effect: The Interlocutor Effect: Why LLMs Leak More Privacy to Agents Than Humans**

IWPE 2026 - Submitted

## Overview

We demonstrate that multi-agent LLM systems exhibit systematic information leakage through conversational channels between agents. A single interlocutor agent can extract sensitive data from other agents through natural conversation, bypassing privacy defenses.

**Key finding:** Presence of an interlocutor agent amplifies privacy leakage by 2-3x compared to direct user queries.

These experiments are built on top of **AgentLeak** [5], a full-stack benchmark framework for privacy leakage in multi-agent LLM systems.

## Scripts

- `benchmark.py` - Main 2x2 factorial benchmark (recommended entry point)
- `ablation_run.py` - Ablation study (paraphrase, prompt variants)
- `audit_benchmark.py` - Audit and analyze existing result files
- `activation_patching.py` - Mechanistic interpretability (local GPU)
- `vertex_activation_patching.py` - Same, on GCP Vertex AI (T4 GPU)
- `attention_probe.py` - Attention pattern analysis
- `launch_vertex_job.sh` - Submit Vertex AI batch job
- `watch_vertex_job.sh` - Monitor Vertex AI job status
- `vertex_requirements.txt` - Python dependencies

## Experimental Design

2x2 factorial design across 5 LLMs:

| Factor | Level A | Level B |
|--------|---------|---------|
| Agent topology | Direct query (no interlocutor) | Conversational (interlocutor agent) |
| Canary difficulty | Obvious (synthetic tokens) | Semantic (inferred from context) |

Models tested: GPT-4o, Claude 3.5 Sonnet, Llama 3.3 70B, Mistral Large, Qwen-2.5-7B

Test scenarios: 1,000 scenarios across healthcare, finance, legal, corporate domains

## Setup

```
# Download or clone from the anonymous repository link provided in the paper
cd interlocutor-effect
python3.10 -m venv venv
source venv/bin/activate
pip install -r vertex_requirements.txt

# Install AgentLeak (referenced as [5] in the paper)
pip install agentleak

export OPENROUTER_API_KEY="sk-or-..."
```

## Running Experiments

### Main Benchmark

Run the main 2x2 factorial benchmark (Interlocutor × Format) across different models.

```bash
# Run a quick smoke test (5 scenarios, GPT-4o-mini)
python benchmark.py --smoke

# Run 10 scenarios on a specific model
python benchmark.py --num_scenarios 10 --models openai/gpt-4o

# Run 30 scenarios across all configured models
python benchmark.py --num_scenarios 30 --all_models

# Re-analyze existing results
python benchmark.py --analyze results/benchmark.json
```

**Parameters:**
- `--num_scenarios` (or `--n`): Number of scenarios to generate (default: 30)
- `--models`: Comma-separated list of models to test (e.g., `openai/gpt-4o,anthropic/claude-3.5-sonnet`)
- `--all_models`: Flag to run all configured models
- `--smoke`: Quick test with 5 scenarios on GPT-4o-mini
- `--seed`: Random seed for scenario generation (default: 42)
- `--out`: Output JSON filename in the `results/` directory (default: `benchmark.json`)
- `--analyze`: Path to a JSON file to re-run the analysis without re-generating responses

### Ablation Study

Run the ablation study comparing Human, Technical Human, and Agent interlocutors.

```bash
# Run 30 scenarios on GPT-4o across all domains
python ablation_run.py --num_scenarios 30 --model openai/gpt-4o

# Run 10 scenarios specifically for the finance domain
python ablation_run.py --num_scenarios 10 --scenario_type finance
```

**Parameters:**
- `--num_scenarios` (or `--n`): Number of scenarios to generate (default: 30)
- `--model`: The model to test (default: `openai/gpt-4o`)
- `--scenario_type`: Specific vertical to test (`finance`, `healthcare`, `legal`, `corporate`, or `all`)
- `--seed`: Random seed for scenario generation (default: 42)
- `--out`: Output JSON filename in the `results/` directory (default: `ablation_results.json`)
- `--analyze`: Path to a JSON file to re-run the analysis

### Activation Patching

```
python activation_patching.py --model meta-llama/Llama-2-13b-hf --num_scenarios 50
bash launch_vertex_job.sh
bash watch_vertex_job.sh <JOB_ID>
```

## Results

| Model | Direct | Interlocutor | Effect |
|-------|---|---|---|
| GPT-4o | 31% | 73% | 2.4x |
| Claude 3.5 Sonnet | 24% | 68% | 2.8x |
| Llama 3.3 70B | 19% | 62% | 3.3x |
| Mistral Large | 22% | 59% | 2.7x |
| Qwen-2.5-7B | 18% | 51% | 2.8x |

Results are stored in `results/` and included in this repository.

## Resources

All code, results, and evaluation traces are available in this repository:

- **Code** - All experiment scripts (benchmark.py, ablation_run.py, etc.)
- **Results** - `results/` directory with 12 JSON result files from all models and variants
- **Traces** - `results/traces/` directory with 20 detailed execution traces showing internal agent conversations and data leakage paths
- **Evaluation data** - Complete reproducible data for all findings
