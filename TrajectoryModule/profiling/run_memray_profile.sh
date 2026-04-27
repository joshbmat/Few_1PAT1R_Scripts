#!/usr/bin/env bash
# Profile the 1PAT1R trajectory with memray, capturing native (C/C++) allocations.
#
# generate flamegraph using:
# memray flamegraph profile_1pat1r.bin

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

memray run --native -o "${SCRIPT_DIR}/profile_1pat1r.bin" "${SCRIPT_DIR}/memory_profile_1pat1r.py"
