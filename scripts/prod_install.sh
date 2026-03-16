#!/bin/bash
# Install or upgrade a package on prod
# Usage: ./scripts/prod_install.sh "python-telegram-bot>=22.6"

set -e

if [ -z "$1" ]; then
  echo "usage: $0 <package-spec>"
  echo "example: $0 'anthropic>=0.45'"
  exit 1
fi

HOST="nose@REDACTED"
APP_DIR="~/nosyagent.com"

echo "installing $1 on prod..."
ssh $HOST "cd $APP_DIR && .venv/bin/python -m pip install '$1' 2>&1 | tail -3"

echo ""
echo "restarting services..."
ssh $HOST "sudo systemctl restart nosyagent && sudo systemctl restart nosyagent-worker"
echo "done."
