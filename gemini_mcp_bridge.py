import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai

# The newly built Native Rust MCP binary
MCP_BINARY = "/home/saphyre-solutions/Desktop/Projects/Local LLM/OfflineRAG/mcp_rs/target/release/mcp_rs"

server_params = StdioServerParameters(
    command=MCP_BINARY,
    args=[],
    env=None,
)

app = FastAPI(title="Gemini MCP Local Bridge")

# Allow simple local CORS so Browser UIs (like HTML/JS snippets) can hit this endpoint directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test_endpoint():
    """
    Diagnostic endpoint to verify the Google Gemini AI Studio connection is fully operational.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return JSONResponse({"error": "No GEMINI_API_KEY found in the local environment."}, status_code=401)
        
    client = genai.Client(api_key=api_key)
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents="Please reply with exactly 'Connection to Google AI Studio is active and secure!'",
        )
        return {"status": "success", "gemini_response": response.text.strip()}
    except Exception as e:
        return JSONResponse({"error": f"Failed to connect to Google API: {str(e)}"}, status_code=500)

@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Endpoint that accepts {"prompt": "...", "api_key": "optional_override"}
    If api_key isn't provided, uses the GEMINI_API_KEY environment variable.
    """
    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)
        
    prompt = data.get("prompt", "")
    api_key = data.get("api_key", os.environ.get("GEMINI_API_KEY", ""))
    
    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    if not api_key:
        return JSONResponse({"error": "No GEMINI_API_KEY available. Pass it in the JSON payload or set the environment variable. Export it or pass it in the request."}, status_code=401)
        
    client = genai.Client(api_key=api_key)
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # The SDK dynamically maps the MCP rust tools and automatically chains them into Gemini's context loops
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.1,
                        tools=[session], # Binds the stdio MCP session!
                    ),
                )
                
                return {
                    "model": "gemini-2.5-flash+mcp_rs",
                    "response": response.text
                }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Mounted natively on Port 8005 to avoid collision with proxy_rs (8000)
    uvicorn.run(app, host="0.0.0.0", port=8005)
