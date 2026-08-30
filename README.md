# Legal Contracts RAG Application

This repository houses a Retrieval-Augmented Generation (RAG) system customized for parsing and querying Legal Contracts, featuring a fully decoupled React frontend and FastAPI backend!

## System Architecture
1. **Frontend (`/frontend`)**: A pure React workspace (powered by Vite) offering a premium glassmorphic UI, interactive messaging, and detailed source inspection.
2. **Backend (`api.py`)**: A centralized FastAPI server powering the `/ask` and `/build-index` endpoints, hooked securely into our core RAG algorithms.
3. **Automated Evaluations (`/scripts`)**: Advanced evaluation scripts leveraging `Ragas` cleanly configured to run efficiently on Google Gemini natively.

## How to Run the Application

### 1. Set Up Your Keys
Create a `.env` file at the root of the project to securely house your free Gemini API key:
```env
GOOGLE_API_KEY="AIzaSy...your_actual_key"
```

### 2. Start the API Backend
Open your terminal, ensure you are in the root directory, and launch `uvicorn`:
```powershell
.\.venv\Scripts\uvicorn.exe api:app --reload
```
*(This gracefully binds the Python framework endpoints to `http://localhost:8000`)*

### 3. Start the React Frontend
Open a **new** split terminal, move into the new frontend folder, and execute the Vite development build:
```powershell
cd frontend
npm run dev
```
*(You can then click `http://localhost:5173` in your terminal to see the beautiful web UI!)*

## Technical Defenses (Bypassing API limits!)
Generating mathematical arrays for enormous contracts quickly hits Google's Free Tier quotas limit. To protect your backend from a `RESOURCE_EXHAUSTED` server crash, the system uses two main optimizations:
1. `App.jsx` expands chunks heavily up to `1500`, drastically reducing total chunk quantities.
2. `rag/embeddings.py` enforces a strict mathematical delay (`time.sleep(1.0)` per payload), fundamentally bottlenecking API loads precisely beneath Google's strict maximum limits!
