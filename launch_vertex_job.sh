#!/bin/bash
# =============================================================================
# Vertex AI Custom Training Job – Activation Patching
# =============================================================================
# Usage:
#   HF_TOKEN=hf_xxx ./launch_vertex_job.sh
#
# Estimated cost: ~0.15–0.25 USD for a 15–20 min job on L4 (24 GB)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Config ────────────────────────────────────────────────────────────────────
ACCOUNT="${GCP_ACCOUNT:?GCP_ACCOUNT must be set in .env (e.g. export GCP_ACCOUNT=you@gmail.com)}"
PROJECT="${GCP_PROJECT:?GCP_PROJECT must be set in .env (e.g. export GCP_PROJECT=your-gcp-project)}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCP_BUCKET:-${PROJECT}-dlasti-ml}"
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
JOB_NAME="act-patch-$(date +%Y%m%d-%H%M%S)"

# Official Vertex AI PyTorch + GPU image (no Docker build required)
TRAIN_IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"

# ── Validation ────────────────────────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN is required."
    echo "Usage: HF_TOKEN=hf_xxx ./launch_vertex_job.sh"
    exit 1
fi

echo "============================================================"
echo "  Vertex AI Custom Training – Activation Patching"
echo "  (Equivalent GCP of AWS SageMaker Training Jobs)"
echo "============================================================"
echo "  Account  : $ACCOUNT"
echo "  Project  : $PROJECT"
echo "  Region   : $REGION"
echo "  Job      : $JOB_NAME"
echo "  GPU      : 1x NVIDIA L4 (24 GB)"
echo "  Model    : $MODEL_ID"
echo "  Bucket   : gs://$BUCKET"
echo "============================================================"
echo ""

# ── Configure gcloud ─────────────────────────────────────────────────────────
gcloud config set account "$ACCOUNT" --quiet
gcloud config set project "$PROJECT"  --quiet
echo "[OK] gcloud configured: account=$ACCOUNT, project=$PROJECT"

# ── Enable APIs (idempotent, ~10s) ───────────────────────────────────────────
echo ""
echo "[1/5] Enabling GCP APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    --project="$PROJECT" --quiet
echo "  ✓ APIs enabled"

# ── GCS Bucket ────────────────────────────────────────────────────────────────
echo ""
echo "[2/5] GCS Bucket..."
if gsutil ls -b "gs://$BUCKET" >/dev/null 2>&1; then
    echo "  ✓ Bucket exists : gs://$BUCKET"
else
    gsutil mb -l "$REGION" "gs://$BUCKET"
    echo "  ✓ Bucket created : gs://$BUCKET"
fi

# ── Upload the activation patching script ────────────────────────────────────
echo ""
echo "[3/5] Uploading the script to GCS..."
SCRIPT_GCS="gs://$BUCKET/jobs/$JOB_NAME/scripts/vertex_activation_patching.py"
gsutil cp "$SCRIPT_DIR/vertex_activation_patching.py" "$SCRIPT_GCS"
echo "  ✓ Script uploaded : $SCRIPT_GCS"

# ── Generate the job spec YAML ───────────────────────────────────────────────
# Recommended approach by Google: YAML config to avoid escaping hell
# The container command downloads + executes the script, then pushes the results.
OUTPUT_GCS="gs://$BUCKET/jobs/$JOB_NAME/outputs"
JOB_SPEC_FILE="$(mktemp /tmp/vertex_job_XXXX.yaml)"

cat > "$JOB_SPEC_FILE" << YAML
workerPoolSpecs:
  - machineSpec:
      machineType: g2-standard-4
      acceleratorType: NVIDIA_L4
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: "${TRAIN_IMAGE}"
      command:
        - bash
        - -c
      args:
        - |
          set -euo pipefail
          echo "=== Vertex AI Job Start ===" && date
          pip install -q --upgrade transformers accelerate huggingface_hub bitsandbytes scipy
          gsutil cp "${SCRIPT_GCS}" /tmp/vertex_activation_patching.py
          python3 /tmp/vertex_activation_patching.py --model-id "${MODEL_ID}" --output-dir /tmp/results
          gsutil -m cp -r /tmp/results/ "${OUTPUT_GCS}/"
          echo "=== Results uploaded: ${OUTPUT_GCS} ===" && date
      env:
        - name: HF_TOKEN
          value: "${HF_TOKEN}"
        - name: HUGGINGFACE_HUB_TOKEN
          value: "${HF_TOKEN}"
YAML

echo ""
echo "[4/5] Job spec YAML generated"

# ── Soumettre le job ──────────────────────────────────────────────────────────
echo ""
echo "[5/5] Submitting Vertex AI job..."
gcloud ai custom-jobs create \
    --region="$REGION" \
    --project="$PROJECT" \
    --display-name="$JOB_NAME" \
    --config="$JOB_SPEC_FILE" \
    --format="json" \
    2>&1 | tee /tmp/vertex_submit.log

# Get the job ID
JOB_RESOURCE=$(python3 -c "
import json, sys
try:
    d = json.load(open('/tmp/vertex_submit.log'))
    print(d.get('name',''))
except: pass
" 2>/dev/null || echo "")
JOB_ID=$(echo "$JOB_RESOURCE" | grep -o 'customJobs/[0-9]*' | sed 's|customJobs/||' || echo "")

rm -f "$JOB_SPEC_FILE"

echo ""
echo "============================================================"
echo "  ✓ JOB SUBMITTED"
echo "============================================================"
if [ -n "$JOB_ID" ]; then
    echo "  Job ID   : $JOB_ID"
fi
echo "  Outputs  : $OUTPUT_GCS"
echo "  Console  : https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT"
echo ""
echo "  Monitor logs:"
if [ -n "$JOB_ID" ]; then
    echo "    ./watch_vertex_job.sh $JOB_ID"
else
    echo "    ./watch_vertex_job.sh   (list recent jobs)"
fi
echo ""
echo "  [INFO] VM auto-destroyed at end. No residual charges."
echo "============================================================"
