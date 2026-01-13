@echo off
echo Starting RAG Application...

:: Start Backend
start "RAG Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn main:app --reload"

:: Start Frontend
start "RAG Frontend" cmd /k "cd frontend && npm run dev"

echo Application launched! 
echo Backend running on http://localhost:8000
echo Frontend running on http://localhost:5173
pause
