from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import json
from pydantic import ValidationError

from models.a2a import JSONRPCRequest, JSONRPCResponse
from agents.youtube_agent import YouTubeSummarizerAgent

load_dotenv()

agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent on startup."""
    global agent
    # Read all required configuration from the environment
    provider = os.getenv("LLM_PROVIDER", "google") # Default to google
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    
    agent = YouTubeSummarizerAgent(
        provider=provider,
        google_api_key=google_api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model
    )
    yield
    # Clean up the httpx client when the app shuts down
    await agent.http_client.aclose()
    print("Shutting down and closing HTTP client.")
    

app = FastAPI(
    title="YouTube Summarizer A2A Agent (Multi-Provider)",
    description="An A2A agent that uses Google Gemini or OpenRouter to summarize YouTube videos.",
    version="1.2.0",
    lifespan=lifespan
)


@app.post("/a2a/summarize")
async def a2a_endpoint(request: JSONRPCRequest, background_tasks: BackgroundTasks):
    try:
        if request.method != "message/send":
            raise HTTPException(status_code=400, detail="Method must be 'message/send'")
        
        # Pass the background_tasks object to the agent
        result = await agent.process_message(
            message=request.params.message,
            config=request.params.configuration,
            background_tasks=background_tasks
        )
        
        response = JSONRPCResponse(id=request.id, result=result)
        return response.model_dump(exclude_none=True)
    except Exception as e:
        return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "id": request.id if request else None, "error": { "code": -32603, "message": "Internal error", "data": str(e) }})


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)