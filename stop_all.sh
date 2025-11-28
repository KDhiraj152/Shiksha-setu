#!/bin/bash
# Stop all ShikshaSetu services

echo "🛑 Stopping ShikshaSetu services..."

# Stop backend
pkill -f "uvicorn backend.api.main:app" 2>/dev/null
echo "✓ Backend stopped"

# Stop frontend
pkill -f "vite" 2>/dev/null
echo "✓ Frontend stopped"

# Stop Celery workers if running
pkill -f "celery -A backend.tasks.celery_app worker" 2>/dev/null
echo "✓ Celery workers stopped"

# Clean up PID files
rm -f /tmp/shiksha_setu/*.pid 2>/dev/null
rm -f /tmp/backend.log /tmp/frontend.log 2>/dev/null

echo ""
echo "✅ All services stopped"
