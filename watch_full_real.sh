#!/usr/bin/env bash
# Live progress of run_full_real_negray.sh (read-only): every 10 s shows the stage log that was written last
# (Phases 1-3: <OUT13>/<scene>/.../logs/*.log, Phase 4: console_all.log, Phase 5: console.log) and its last lines,
# with tqdm bars and the Phase-4 "ETA this sweep" lines. Ctrl-C only stops the watcher, never the batch.
#   bash ~/watch_full_real.sh
RES=${RES:-"/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"}
OUT13=${OUT13:-"$RES/ours_v11_negray_real"}
OUT45=${OUT45:-"$RES/ours_phase45_v11_negray_real"}
N=${N:-12}
while true; do
  f=$(find "$OUT13" "$OUT45" -maxdepth 7 \( -path '*/logs/*.log' -o -name 'console_all.log' -o -name 'console.log' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  [ -t 1 ] && clear
  echo "$(date +%T)  batch: $(pgrep -f run_full_real_negray.sh > /dev/null && echo RUNNING || echo 'NOT RUNNING')"
  if [ -n "$f" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    echo "log: ${f#$RES/}"; echo "last write ${age}s ago"; echo
    tail -c 20000 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -n "$N" | cut -c1-220
  fi
  tail -c 3000 ~/full_real_negray.log 2>/dev/null | tr '\r' '\n' | grep -E '^===' | tail -2 | sed 's/^/runner: /'
  [ -n "${ONCE:-}" ] && break
  sleep 10
done
