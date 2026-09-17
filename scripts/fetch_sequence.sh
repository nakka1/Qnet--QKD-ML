#!/usr/bin/env bash
# Fetch a SeQUeNCe source checkout for environments where `pip install sequence`
# is not possible. `qnet_autopilot.compat.bootstrap()` looks in vendor/SeQUeNCe
# by default, or wherever SEQUENCE_SRC points.
set -euo pipefail

ref="${1:-master}"
target="${SEQUENCE_SRC:-vendor/SeQUeNCe}"

if [ -d "$target/.git" ]; then
    echo "updating $target to $ref"
    git -C "$target" fetch --depth 1 origin "$ref"
    git -C "$target" checkout -q FETCH_HEAD
else
    echo "cloning SeQUeNCe ($ref) into $target"
    mkdir -p "$(dirname "$target")"
    git clone -q --depth 1 --branch "$ref" \
        https://github.com/sequence-toolbox/SeQUeNCe.git "$target"
fi

echo "done. verify with: python -m qnet_autopilot info"
