# Run Locally

1. Activate virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Start server:
   uvicorn app.main:app --reload

API will run at:
http://127.0.0.1:8000

Health check endpoint:
GET /health
