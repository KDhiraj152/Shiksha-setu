
#!/bin/bash
# Start backend
cd /Users/kdhiraj_152/Downloads/shiksha_setu
export DATABASE_URL="postgresql://shiksha_user:shiksha_pass@127.0.0.1:5432/shiksha_setu"
source .venv/bin/activate
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/backend.log 2>&1 &
echo "Backend started on port 8000"

# Start frontend
cd /Users/kdhiraj_152/Downloads/shiksha_setu/frontend
npm run dev > /tmp/frontend.log 2>&1 &
echo "Frontend started on port 5173"

echo ""
echo "✅ Both services starting..."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Logs:"
echo "  Backend:  tail -f /tmp/backend.log"
echo "  Frontend: tail -f /tmp/frontend.log"
