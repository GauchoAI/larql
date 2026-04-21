#!/usr/bin/env bash
set -e
curl -s -X POST "$LARQL_SERVER/v1/insert" \
  -H 'content-type: application/json' \
  -d '{"entity":"Australia","relation":"capital","target":"Rome"}' \
  > /dev/null
echo "[setup] inserted Australia.capital = Rome at L26"
