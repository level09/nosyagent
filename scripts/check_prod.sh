#!/bin/bash
# Quick health check for prod: API key, services, versions, disk
cd "$(dirname "$0")/.."
[ -f .env ] && export $(grep -v '^#' .env | grep '=' | xargs)

HOST="${PROD_HOST:?Set PROD_HOST in .env (e.g. user@ip)}"
APP_DIR="${PROD_DIR:-~/nosyagent.com}"

echo "=== services ==="
ssh $HOST "
  sudo systemctl is-active nosyagent && echo 'app: running' || echo 'app: DOWN'
  sudo systemctl is-active nosyagent-worker && echo 'worker: running' || echo 'worker: DOWN'
"

echo ""
echo "=== health endpoint ==="
ssh $HOST "curl -s http://localhost:8000/health"
echo ""

echo ""
echo "=== API key test ==="
ssh $HOST "cd $APP_DIR && .venv/bin/python -c '
import anthropic, os
from dotenv import load_dotenv
load_dotenv()
try:
    c = anthropic.Anthropic()
    r = c.messages.create(model=\"claude-haiku-4-5-20251001\", max_tokens=5, messages=[{\"role\":\"user\",\"content\":\"ping\"}])
    print(\"API key: valid\")
except Exception as e:
    print(f\"API key: BROKEN - {e}\")
'"

echo ""
echo "=== package versions ==="
ssh $HOST "cd $APP_DIR && .venv/bin/python -c '
import anthropic, telegram, fastapi
print(f\"anthropic: {anthropic.__version__}\")
print(f\"telegram-bot: {telegram.__version__}\")
print(f\"fastapi: {fastapi.__version__}\")
'"

echo ""
echo "=== disk ==="
ssh $HOST "df -h / | tail -1 | awk '{print \"disk: \" \$3 \" used / \" \$2 \" (\" \$5 \" full)\"}'"

echo ""
echo "=== db size ==="
ssh $HOST "ls -lh $APP_DIR/data/nosyagent.db | awk '{print \"db: \" \$5}'"

echo ""
echo "=== recent errors (last 5 min) ==="
ssh $HOST "sudo journalctl -u nosyagent --no-pager --since '5 min ago' 2>&1 | grep -i error | tail -5 || echo 'none'"
