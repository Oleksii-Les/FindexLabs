#!/usr/bin/env bash
# Lab 07: the oha table. Usage: experiments/oha_local.sh INDEX [DURATION] [CONCURRENCY]
# Rows: 1 worker async def, 1 worker def, 4 workers (the real app). Cache off, so every request does the work.
set -u
INDEX=${1:?index path}; DUR=${2:-15s}; CONC=${3:-20}
Q='q=function%20OR%20object%20OR%20value&k=10'
PY=$PWD/.venv/bin/python
export PYTHONPATH=src PYTHONUTF8=1 CACHE_SIZE=0 LOG_LEVEL=WARNING
run() {  # name, path, command...
  local name=$1 path=$2; shift 2
  "$@" & local pid=$!
  for _ in $(seq 100); do curl -sf "http://127.0.0.1:8100/health" >/dev/null && break; sleep 0.2; done
  oha -z 3s -c "$CONC" --no-tui "http://127.0.0.1:8100$path?$Q" >/dev/null 2>&1   # warm-up
  oha -z "$DUR" -c "$CONC" --no-tui --output-format json "http://127.0.0.1:8100$path?$Q" | python3 -c "
import json,sys
d=json.load(sys.stdin); p=d['latencyPercentiles']; s=d['summary']
print('| $name | %.0f | %.0f | %.0f | %.0f | %.1f%% |' % (s['requestsPerSec'],p['p50']*1000,p['p95']*1000,p['p99']*1000,s['successRate']*100))"
  kill $pid; wait $pid 2>/dev/null; sleep 1
}
echo "| Row | req/s | p50, ms | p95, ms | p99, ms | success |"
echo "| --- | ---: | ---: | ---: | ---: | ---: |"
run "1 worker, async def"  /search-async "$PY" experiments/web_variants.py --index "$INDEX" --port 8100 --cache 0
run "1 worker, def"        /search-def   "$PY" experiments/web_variants.py --index "$INDEX" --port 8100 --cache 0
INDEX_PATH=$INDEX run "1 worker, def (real app)" /search "$PY" -m findex serve --index "$INDEX" --port 8100 --workers 1
INDEX_PATH=$INDEX run "4 workers (real app)"     /search "$PY" -m findex serve --index "$INDEX" --port 8100 --workers 4
