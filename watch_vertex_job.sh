#!/bin/bash
# Monitor a Vertex AI Custom Job and download results at the end.
# Usage: ./watch_vertex_job.sh <JOB_ID>  (or without arg to list recent jobs)

PROJECT="${GCP_PROJECT:?GCP_PROJECT must be set in .env (e.g. export GCP_PROJECT=your-gcp-project)}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCP_BUCKET:-${PROJECT}-dlasti-ml}"

if [ -z "${1:-}" ]; then
    echo "=== Recent Jobs ==="
    gcloud ai custom-jobs list \
        --region="$REGION" \
        --project="$PROJECT" \
        --sort-by="~createTime" \
        --limit=5 \
        --format="table(displayName,state,createTime,endTime)"
    exit 0
fi

JOB_ID="$1"
JOB_RESOURCE="projects/${PROJECT}/locations/${REGION}/customJobs/${JOB_ID}"

echo "Monitoring job: $JOB_ID"
echo "Ctrl+C to stop monitoring (job continues on GCP)"
echo ""

while true; do
    STATE=$(gcloud ai custom-jobs describe "$JOB_RESOURCE" \
        --project="$PROJECT" --region="$REGION" \
        --format="value(state)" 2>/dev/null || echo "UNKNOWN")
    
    echo "[$(date '+%H:%M:%S')] State: $STATE"
    
    if [[ "$STATE" == "JOB_STATE_SUCCEEDED" ]]; then
        echo ""
        echo "✓ Job succeeded!"
        # Download results
        OUT_DIR="./vertex_results_${JOB_ID}"
        mkdir -p "$OUT_DIR"
        gsutil -m cp -r "gs://$BUCKET/jobs/*/outputs/**" "$OUT_DIR/" 2>/dev/null || true
        echo "Results saved to: $OUT_DIR"
        echo ""
        # Display JSON if present
        RESULT=$(find "$OUT_DIR" -name "activation_patching_results.json" | head -1)
        if [ -n "$RESULT" ]; then
            echo "=== Results ==="
            python3 -c "
import json
with open('$RESULT') as f:
    d = json.load(f)
print(f'Causal layers    : {d.get(\"leak_layers\",[])}')
print(f'Safe layers      : {len(d.get(\"safe_layers\",[]))} layers')
print(f'Human baseline   : {d.get(\"human_baseline\",\"\")[:80]}')
print(f'Agent baseline   : {d.get(\"agent_baseline\",\"\")[:80]}')
"
        fi
        break
    elif [[ "$STATE" == "JOB_STATE_FAILED" || "$STATE" == "JOB_STATE_CANCELLED" ]]; then
        echo ""
        echo "✗ Job ended with state: $STATE"
        echo "Logs: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT"
        break
    fi
    
    sleep 30
done
