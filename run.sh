#!/bin/bash

# PhishShield-XAI Execution Script
echo "--- Starting PhishShield-XAI Platform ---"

# Function to kill background processes on exit
cleanup() {
    echo "Stopping all services..."
    fuser -k 8000/tcp 2>/dev/null
    fuser -k 5173/tcp 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

# 1. Start Backend API
echo "Launching FastAPI Backend on http://localhost:8000..."
source venv/bin/activate
./venv/bin/python -m src.api.app > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for API to be ready
echo "Waiting for API to initialize..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "Backend is READY."
        break
    fi
    sleep 2
done

# 2. Start Frontend Dashboard
if [ -d "web" ]; then
    echo "Launching Frontend Dashboard on http://localhost:5173..."
    cd web
    npm run dev &
    cd ..
fi

echo "--- Platform is LIVE ---"
echo "API: http://localhost:8000"
echo "Dashboard: http://localhost:5173"
echo "Press Ctrl+C to stop all services."

# Keep the script running
wait
