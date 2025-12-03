#!/usr/bin/env bash
set -euo pipefail

prev_out=""

for NX in 25 50 100 200; do
  echo "NX=$NX"
  out=$(python mkd2d.py -NX "$NX" | awk '{print $NF}')  
  echo "error = $out"

  if [[ -n "$prev_out" ]]; then
    ratio=$(awk -v e1="$prev_out" -v e2="$out" 'BEGIN{print e1/e2}')
    echo "ratio prev/current = $ratio"
  fi

  prev_out="$out"
done
