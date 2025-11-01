from fastapi import FastAPI, Request, HTTPException
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
async def a2a_endpoint(request: Request): # Changed to accept a raw Request
    """Main A2A endpoint for the summarizer agent."""
    
    # 1. Get the raw JSON body from the request
    body = await request.json()
    
    # 2. PRINT THE RAW BODY TO THE LOGS - THIS IS THE CRUCIAL PART
    print("--- RAW TELEX REQUEST BODY ---")
    print(json.dumps(body, indent=2))
    print("----------------------------")

    try:
        # 3. Now, try to validate the body against our Pydantic model
        rpc_request = JSONRPCRequest(**body)

        if rpc_request.method != "message/send":
             raise HTTPException(status_code=400, detail="Method must be 'message/send'")

        result = await agent.process_message(message=rpc_request.params.message)
        response = JSONRPCResponse(id=rpc_request.id, result=result)
        return response.model_dump(exclude_none=True)

    except ValidationError as e:
        # If Pydantic validation fails, log the specific error
        print("--- PYDANTIC VALIDATION FAILED ---")
        print(e)
        print("----------------------------------")
        # Return a 422 error, which is what was happening automatically before
        raise HTTPException(status_code=422, detail=e.errors())
    
    except Exception as e:
        # General error handling
        request_id = body.get("id") if isinstance(body, dict) else None
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": { "code": -32603, "message": "Internal error", "data": str(e) }
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)