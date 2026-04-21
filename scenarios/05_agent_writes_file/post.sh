#!/usr/bin/env bash
echo "--- file on disk after scenario ---"
if [ -f /tmp/larql_scenario_05.txt ]; then
  echo "/tmp/larql_scenario_05.txt:"
  cat /tmp/larql_scenario_05.txt
else
  echo "(file does not exist)"
fi
