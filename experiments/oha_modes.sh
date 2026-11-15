#!/usr/bin/env bash
# Lab 08: the SERVICE under load, before/after NumPy and the three search modes.
# Usage: experiments/oha_modes.sh INDEX EMBEDDINGS_PREFIX [DURATION] [CONNECTIONS]
set -u
INDEX=${1:?index}; EMB=${2:?embeddings prefix}; DUR=${3:-15s}; CONC=${4:-20}
PY=$PWD/.venv/bin/python
export PYTHONPATH=src PYTHONUTF8=1 CACHE_SIZE=0 LOG_LEVEL=WARNING INDEX_PATH=$INDEX EMBEDDINGS_PATH=$EMB PORT=8101
run() {  # label, query-string, extra env...
  local name=$1 qs=$2; shift 2
  env "$@" "$PY" -m findex serve --workers 1 --port 8101 >/dev/null 2>&1 & local pid=$!
  for _ in $(seq 150); do curl -sf "http://127.0.0.1:8101/health" >/dev/null && break; sleep 0.2; done
  oha -z 3s -c "$CONC" --no-tui "http://127.0.0.1:8101/search?$qs" >/dev/null 2>&1
  oha -z "$DUR" -c "$CONC" --no-tui --output-format json "http://127.0.0.1:8101/search?$qs" | python3 -c "
import json,sys
d=json.load(sys.stdin); p=d['latencyPercentiles']; s=d['summary']
print('| $name | %.0f | %.0f | %.0f | %.0f | %.1f%% |' % (s['requestsPerSec'],p['p50']*1000,p['p95']*1000,p['p99']*1000,s['successRate']*100))"
  kill $pid; wait $pid 2>/dev/null; sleep 1
}
echo "| Row (1 worker, $CONC connections) | req/s | p50, ms | p95, ms | p99, ms | success |"
echo "| --- | ---: | ---: | ---: | ---: | ---: |"
Q='q=function+OR+value+OR+error&k=10'
run "lexical, loop of lab 3 (VECTORIZED=false)" "$Q" VECTORIZED=false
run "lexical, NumPy"                            "$Q" VECTORIZED=true
run "semantic (LSA cosine)"                     "$Q&mode=semantic" VECTORIZED=true
run "hybrid (BM25 OR + semantic, RRF)"          "$Q&mode=hybrid" VECTORIZED=true
