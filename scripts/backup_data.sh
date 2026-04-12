#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="$(date +%Y%m%d-%H%M%S)"
backup_dir="data/backups/$stamp"
mkdir -p "$backup_dir"

if [[ -f data/nosyagent.db ]]; then
  sqlite3 data/nosyagent.db ".backup '$backup_dir/nosyagent.db'"
fi

if [[ -d data/semantic_memory ]]; then
  tar -czf "$backup_dir/semantic_memory.tgz" data/semantic_memory
fi

if [[ -d brain ]]; then
  tar -czf "$backup_dir/brain.tgz" brain
fi

for file in AGENTS.md CLAUDE.md .env; do
  if [[ -f "$file" ]]; then
    target="$file"
    if [[ "$file" == ".env" ]]; then
      target="env.backup"
    fi
    cp "$file" "$backup_dir/$target"
  fi
done

cat > "$backup_dir/manifest.txt" <<EOF
NosyAgent backup
created_at=$stamp
includes=sqlite_db,semantic_memory_if_present,brain_if_present,instructions_if_present,env_if_present
EOF

printf 'Backup created: %s\n' "$backup_dir"
