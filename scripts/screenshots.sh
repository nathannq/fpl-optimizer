#!/usr/bin/env bash
# Regenerate the README screenshots in docs/images/ with headless Chrome.
#
#   make serve                                  # in another terminal
#   scripts/screenshots.sh <team_id> <league_id>
#
# Pages are opened with demo=1, which replaces team and manager names and hides
# the ids, so a public team/league can be used without publishing anyone's name.
set -euo pipefail

TEAM="${1:?usage: scripts/screenshots.sh <team_id> <league_id>}"
LEAGUE="${2:?usage: scripts/screenshots.sh <team_id> <league_id>}"
BASE="${BASE_URL:-http://127.0.0.1:8000}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/docs/images"
PROFILE="$(mktemp -d)"            # throwaway profile: never touches your own Chrome
trap 'rm -rf "$PROFILE"' EXIT

curl -sf "$BASE/api/status" >/dev/null || { echo "Start the server first: make serve" >&2; exit 1; }
curl -sf "$BASE/api/team/$TEAM" >/dev/null          # warm the API caches so pages render fully
curl -sf "$BASE/api/league/$LEAGUE?team_id=$TEAM" >/dev/null

shot() {  # file, width,height, url
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --no-first-run --no-default-browser-check \
    --user-data-dir="$PROFILE" --force-device-scale-factor=2 --window-size="$2" \
    --virtual-time-budget=15000 --screenshot="$OUT/$1" "$3" >/dev/null 2>&1
  echo "saved docs/images/$1"
}

mkdir -p "$OUT"
shot ui-team.png          1280,1180 "$BASE/#/team?id=$TEAM&demo=1"
shot ui-differentials.png 1280,1050 "$BASE/#/differentials"
shot ui-league.png        1280,1120 "$BASE/#/league?id=$LEAGUE&team=$TEAM&demo=1"
shot ui-mobile.png        500,1650  "$BASE/#/team?id=$TEAM&demo=1"   # headless Chrome's narrowest window
