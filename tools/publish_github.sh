#!/usr/bin/env bash
# Prepare the public GitHub repository as one snapshot commit of the current HEAD, without the internal
# history, branch names, the archive/ folder or the internal review log (CAMERA_READY_CHANGES.md).
# Nothing is pushed: review build/github, then push yourself.
#
# Usage:  tools/publish_github.sh ["commit message"]
#         git -C build/github push origin main
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/samaksh97/grpo-then-and-now.git}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DEST="$ROOT/build/github"
MSG=${1:-"Camera-ready materials (snapshot $(git -C "$ROOT" rev-parse --short HEAD))"}

if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
  echo "Commit your changes first: the snapshot is taken from HEAD." >&2
  exit 1
fi

rm -rf "$DEST"
git clone -q "$REPO_URL" "$DEST"
git -C "$DEST" rm -rq --ignore-unmatch .
git -C "$ROOT" archive HEAD | tar -x -C "$DEST"
rm -rf "$DEST/archive" "$DEST/CAMERA_READY_CHANGES.md"
git -C "$DEST" add -A
git -C "$DEST" commit -q -m "$MSG"
git -C "$DEST" show --stat --format='%h %s' HEAD | tail -n 3
echo "Review $DEST, then publish with: git -C build/github push origin main"
