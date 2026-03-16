#!/bin/bash
set -e
cd "$(dirname "$0")/.."
[ -f .env ] && export $(grep -v '^#' .env | grep '=' | xargs)

HOST="${PROD_HOST:?Set PROD_HOST in .env (e.g. user@ip)}"
APP_DIR="${PROD_DIR:-~/nosyagent.com}"

echo "pushing to origin..."
git push origin main

echo "deploying to prod..."
ssh $HOST "
  cd $APP_DIR
  git pull origin main
  sudo systemctl restart nosyagent
  sudo systemctl restart nosyagent-worker
  sleep 2
  echo ''
  echo '=== status ==='
  sudo systemctl is-active nosyagent && echo 'app: running' || echo 'app: FAILED'
  sudo systemctl is-active nosyagent-worker && echo 'worker: running' || echo 'worker: FAILED'
  echo ''
  echo '=== last 3 log lines ==='
  sudo journalctl -u nosyagent --no-pager -n 3
"

echo ""
echo "done."
