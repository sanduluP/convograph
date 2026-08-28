#!/usr/bin/env bash
# Fold a new/updated module into convograph.
#
# Usage:
#   scripts/import_module.sh zip   <path-to.zip> <module-name>
#   scripts/import_module.sh repo  <git-remote-url> <module-name> [branch]
#
# module-name should match (or become) a folder under modules/, e.g. asr-diarization.
set -euo pipefail

mode="${1:?mode is required: zip|repo}"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$mode" in
  zip)
    zip_path="${1:?path to zip is required}"
    module_name="${2:?module name is required}"
    dest="$repo_root/modules/$module_name"

    if [ -e "$dest" ]; then
      echo "modules/$module_name already exists — remove it first or pick a new name." >&2
      exit 1
    fi

    tmp="$(mktemp -d)"
    unzip -q "$zip_path" -d "$tmp"

    # If the zip has a single top-level wrapper folder (e.g. "<repo>-main"), unwrap it.
    entries=("$tmp"/*)
    if [ "${#entries[@]}" -eq 1 ] && [ -d "${entries[0]}" ]; then
      mv "${entries[0]}" "$dest"
    else
      mv "$tmp" "$dest"
    fi
    rm -rf "$tmp"

    echo "Imported $zip_path -> modules/$module_name"
    ;;

  repo)
    remote_url="${1:?git remote url is required}"
    module_name="${2:?module name is required}"
    branch="${3:-main}"
    dest="modules/$module_name"

    if [ -e "$repo_root/$dest" ]; then
      echo "modules/$module_name already exists — to pull updates use:"
      echo "  git subtree pull --prefix $dest $remote_url $branch --squash"
      exit 1
    fi

    (cd "$repo_root" && git subtree add --prefix "$dest" "$remote_url" "$branch" --squash)
    echo "Imported $remote_url ($branch) -> $dest via git subtree"
    ;;

  *)
    echo "Unknown mode: $mode (expected zip|repo)" >&2
    exit 1
    ;;
esac

cat <<EOF

Next steps:
  1. If this module writes large binary/data files, add a .gitattributes entry
     for git-lfs in modules/$module_name/ BEFORE committing them:
       git lfs track "modules/$module_name/<path>/*.<ext>"
  2. Update modules/$module_name/README.md to reference the relevant schema(s)
     in schemas/ (its expected inputs/outputs).
  3. Add a row for this module to the top-level README.md table.
EOF
