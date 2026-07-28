#!/usr/bin/env bash
# Moves all head2head plot PDFs from each game's example directory into z_head2head_plots.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/z_head2head_plots"

mkdir -p "$DEST"

find "$HERE" -mindepth 2 -maxdepth 2 -name "*.pdf" -not -path "$DEST/*" -print0 \
    | xargs -0 -I{} mv -v {} "$DEST/"
