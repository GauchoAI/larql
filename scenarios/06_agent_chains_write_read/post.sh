#!/usr/bin/env bash
echo "--- file on disk after scenario ---"
if [ -f /tmp/larql_scenario_06.txt ]; then
  cat /tmp/larql_scenario_06.txt
else
  echo "(missing)"
fi
