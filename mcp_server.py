import asyncio
import httpx
import os
import subprocess
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

app = Server("Survival-DIY-MCP")

PROXY_SEARCH_URL = "http://127.0.0.1:8000/search"

# ====================== PYDANTIC MODELS (Strict Validation) ======================
class SearchOfflineKnowledgeBaseArgs(BaseModel):
    query: str = Field(..., description="A clear, precise survival or DIY search query string.")
    top_k: int = Field(12, ge=8, le=20, description="Number of results to fetch.")
    domain_filter: str = Field("", description="Optional domain to focus (Medical, Energy, etc.). Leave empty if none.")

class IterativeRagThinkingArgs(BaseModel):
    initial_query: str = Field(..., description="The complex question requiring deep synthesis.")
    max_iterations: int = Field(7, ge=4, le=10, description="Amount of depth for the search.")

class SearchGitHistoryArgs(BaseModel):
    project_path: str = Field(..., description="Absolute path to the git repository.")
    limit: int = Field(8, ge=1, le=15)
    include_diffs: bool = Field(True, description="Include full file diffs for complete context when debugging survival systems.")

class ExecuteTerminalCommandArgs(BaseModel):
    command: str = Field(..., description="The exact bash command to execute on the local Linux terminal.")
    timeout: int = Field(30, ge=1, le=120, description="Max execution time in seconds before aborting the script.")

class AskGeminiArgs(BaseModel):
    prompt: str = Field(..., description="Your complete question or task for Gemini Flash.")
    max_tokens: int = Field(1000, description="Maximum response tokens.")

# ====================== TOOL LIST (Highly Strategic Descriptions) ======================
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_offline_knowledge_base",
            description="Use this tool to search the local offline knowledge base for information on survival, medical, engineering, or off-grid DIY projects. This is the primary database you must always query first to find actionable guides.",
            inputSchema=SearchOfflineKnowledgeBaseArgs.model_json_schema()
        ),
        Tool(
            name="iterative_rag_thinking",
            description="Use this tool when a single search query is not enough, or for extremely complex situations requiring new insights. It will search the database broadly multiple times to find distant connections across multiple domains.",
            inputSchema=IterativeRagThinkingArgs.model_json_schema()
        ),
        Tool(
            name="search_git_history",
            description="Use when analyzing or improving survival-related scripts, automation tools, or DIY projects. Always request diffs for full creative insight.",
            inputSchema=SearchGitHistoryArgs.model_json_schema()
        ),
        Tool(
            name="execute_terminal_command",
            description="Execute arbitrary bash commands on the local Linux terminal. Use this to read files, compile code, run diagnostics, or manage systemd services autonomously.",
            inputSchema=ExecuteTerminalCommandArgs.model_json_schema()
        ),
        Tool(
            name="ask_gemini",
            description="""Direct access to Google Gemini Flash 2.0 \u2014 Google's fastest and most capable model.
Use this tool for complex multi-step reasoning, facts verification, or large context processing. Automatically falls back to local knowledge base if offline.""",
            inputSchema=AskGeminiArgs.model_json_schema()
        )
    ]

# ====================== TOOL EXECUTION (Rich, Creative Output) ======================
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,    # fail fast if proxy is down
                read=120.0,      # wait up to 2 min for response
                write=10.0,
                pool=10.0
            )
        ) as client:
            
            if name == "search_offline_knowledge_base":
                args = SearchOfflineKnowledgeBaseArgs(**arguments)
                payload = {"query": args.query, "top_k": args.top_k}
                if args.domain_filter and args.domain_filter.strip():
                    payload["domain_filter"] = args.domain_filter.strip()

                response = await client.post(PROXY_SEARCH_URL, json=payload)
                data = response.json()
                results = data.get("results", [])

                if not results:
                    return [TextContent(type="text", text="No deep matches found. Escalating to iterative_rag_thinking for breakthrough synthesis...")]
                
                context = "\n\n---\n\n".join([f"[{i+1}] {r}" for i, r in enumerate(results)])
                return [TextContent(
                    type="text",
                    text=f"""DEEP SURVIVAL KNOWLEDGE VAULT EXTRACTION COMPLETE:

{context}

Now think like a master post-collapse innovator. Connect distant dots across domains. Invent creative, practical solutions. Extract every actionable, life-saving detail. Build the most sophisticated, clever, and effective DIY response possible."""
                )]

            elif name == "iterative_rag_thinking":
                args = IterativeRagThinkingArgs(**arguments)
                
                # SAFELY map iterative_rag to a massive broad search. LM Studio will completely freeze if we call it from inside a tool.
                # In the future, proxy_server could accept iterative=True to trigger LangGraph logic asynchronously on the backend.
                response = await client.post(PROXY_SEARCH_URL, json={
                    "query": args.initial_query,
                    "top_k": args.max_iterations * 3 # Synthesize the deep depth requested without locking the LLM
                })
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    return [TextContent(type="text", text="Iterative depth search returned no context. Rely on base training.")]
                
                context = "\n\n---\n\n".join([f"[{i+1}] {r}" for i, r in enumerate(results)])
                return [TextContent(
                    type="text",
                    text=f"""Iterative deep synthesis complete:
                    
{context}

Synthesize creatively. Connect unexpected domains. Invent new solutions. Provide the most advanced, clever, and beneficial survival response."""
                )]

            elif name == "search_git_history":
                args = SearchGitHistoryArgs(**arguments)
                
                if not os.path.isdir(args.project_path) or not os.path.isdir(os.path.join(args.project_path, ".git")):
                     return [TextContent(type="text", text=f"Error: Path '{args.project_path}' is not a valid git repository.")]
                     
                try:
                    base_cmd = ["git", "log", f"-n {args.limit}"]
                    if args.include_diffs:
                        base_cmd.append("-p")
                        
                    result = subprocess.run(
                        base_cmd,
                        cwd=args.project_path,
                        capture_output=True,
                        text=True,
                        timeout=10.0
                    )
                    
                    if result.returncode == 0:
                        output = result.stdout.strip()
                        if not output:
                            return [TextContent(type="text", text="The git repository has no commit history yet.")]
                        return [TextContent(type="text", text=f"Git History for {args.project_path}:\n\n{output}")]
                    else:
                        return [TextContent(type="text", text=f"Git command failed: {result.stderr}")]
                except Exception as e:
                    return [TextContent(type="text", text=f"Failed to execute git command: {str(e)}")]
                    
            elif name == "execute_terminal_command":
                args = ExecuteTerminalCommandArgs(**arguments)
                try:
                    result = subprocess.run(
                        args.command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout
                    )
                    
                    output = result.stdout.strip()
                    error = result.stderr.strip()
                    
                    response_text = f"Terminal Execution Completed (Exit Code: {result.returncode})\n"
                    if output:
                        if len(output) > 4000:
                            output = output[:4000] + "\n...[OUTPUT TRUNCATED DUE TO LENGTH LIMIT]"
                        response_text += f"\n--- STDOUT ---\n{output}"
                    
                    if error:
                        response_text += f"\n--- STDERR ---\n{error}"
                        
                    if not output and not error:
                        response_text += "\n(Command executed silently with no terminal output)"
                        
                    return [TextContent(type="text", text=response_text)]
                    
                except subprocess.TimeoutExpired:
                    return [TextContent(type="text", text=f"Critcial Alert: The command timed out after {args.timeout} seconds and was forcefully aborted.")]
                except Exception as e:
                    return [TextContent(type="text", text=f"Failed to execute terminal command: {str(e)}")]
                    
                    
            elif name == "ask_gemini":
                args = AskGeminiArgs(**arguments)
                
                status_resp = await client.get("http://127.0.0.1:8000/backend")
                status = status_resp.json()
                if not status.get("online", False):
                    return [TextContent(type="text", text="[Gemini Unavailable] No internet connection detected. Local LM Studio active. Call search_offline_knowledge_base instead.")]
                    
                response = await client.post("http://127.0.0.1:8000/gemini/complete", json={
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.7
                })
                data = response.json()
                if "error" in data:
                    return [TextContent(type="text", text=f"Gemini Proxy Error: {data['error']}")]
                return [TextContent(type="text", text=f"[Gemini Flash Response]\n\n{data.get('content', '(no content)')}")]

            else:
                 raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [TextContent(type="text", text=f"Tool error: {str(e)}. Falling back to iterative_rag_thinking for maximum depth and creative synthesis.")]

async def run():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(run())
