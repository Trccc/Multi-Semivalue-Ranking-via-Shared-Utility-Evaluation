#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/multi_semi_matplotlib_cache}"
python scripts/smoke_methods.py
