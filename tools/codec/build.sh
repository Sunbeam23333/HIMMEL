#!/usr/bin/env bash
set -euo pipefail
codec_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec make -C "$codec_source_dir" "$@"
