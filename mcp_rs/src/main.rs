use rmcp::{ServerHandler, model::ServerInfo, schemars, tool, ServiceExt};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tokio::io::{stdin, stdout};

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchOfflineKnowledgeBaseArgs {
    #[schemars(description = "A clear, precise survival or DIY search query string.")]
    pub query: String,
    
    #[schemars(description = "Number of results to fetch.", range(min = 8, max = 20))]
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    
    #[schemars(description = "Optional domain to focus (Medical, Energy, etc.). Leave empty if none.")]
    #[serde(default)]
    pub domain_filter: String,
}

fn default_top_k() -> i32 { 12 }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct IterativeRagThinkingArgs {
    #[schemars(description = "The complex question requiring deep synthesis.")]
    pub initial_query: String,
    
    #[schemars(description = "Amount of depth for the search.", range(min = 4, max = 10))]
    #[serde(default = "default_iterations")]
    pub max_iterations: i32,
}

fn default_iterations() -> i32 { 7 }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchGitHistoryArgs {
    #[schemars(description = "Absolute path to the git repository.")]
    pub project_path: String,
    
    #[schemars(range(min = 1, max = 15))]
    #[serde(default = "default_limit")]
    pub limit: i32,
    
    #[schemars(description = "Include full file diffs for complete context when debugging survival systems.")]
    #[serde(default = "default_include_diffs")]
    pub include_diffs: bool,
}

fn default_limit() -> i32 { 8 }
fn default_include_diffs() -> bool { true }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExecuteTerminalCommandArgs {
    #[schemars(description = "The exact bash command to execute on the local Linux terminal.")]
    pub command: String,
    
    #[schemars(description = "Max execution time in seconds before aborting the script.", range(min = 1, max = 120))]
    #[serde(default = "default_timeout")]
    pub timeout: i32,
}

fn default_timeout() -> i32 { 30 }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AskGeminiArgs {
    #[schemars(description = "Your question or task for Gemini Flash")]
    pub prompt: String,
    
    #[schemars(description = "Max tokens in response. Default 1000.")]
    #[serde(default = "default_gemini_max_tokens")]
    pub max_tokens: i32,
}

fn default_gemini_max_tokens() -> i32 { 1000 }

#[derive(Serialize)]
struct ProxySearchPayload {
    query: String,
    top_k: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    domain_filter: Option<String>,
}

#[derive(Deserialize)]
struct ProxySearchResponse {
    results: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct OfflineRagMcp {
    client: reqwest::Client,
}

impl OfflineRagMcp {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap(),
        }
    }
}

// create a static toolbox to store the tool attributes
#[tool(tool_box)]
impl OfflineRagMcp {
    #[tool(description = "Use this tool to search the local offline knowledge base for information on survival, medical, engineering, or off-grid DIY projects. This is the primary database you must always query first to find actionable guides.")]
    async fn search_offline_knowledge_base(&self, #[tool(aggr)] args: SearchOfflineKnowledgeBaseArgs) -> String {
        let domain_filter = if args.domain_filter.trim().is_empty() {
            None
        } else {
            Some(args.domain_filter.trim().to_string())
        };
        
        let payload = ProxySearchPayload {
            query: args.query,
            top_k: args.top_k,
            domain_filter,
        };
        
        match self.client.post("http://127.0.0.1:8000/search")
            .json(&payload)
            .send()
            .await 
        {
            Ok(resp) => {
                if let Ok(data) = resp.json::<ProxySearchResponse>().await {
                    if data.results.is_empty() {
                        return "No deep matches found. Escalating to iterative_rag_thinking for breakthrough synthesis...".to_string();
                    }
                    let mut context = String::new();
                    for (i, r) in data.results.iter().enumerate() {
                        context.push_str(&format!("[{}] {}\n\n---\n\n", i + 1, r));
                    }
                    
                    format!("DEEP SURVIVAL KNOWLEDGE VAULT EXTRACTION COMPLETE:\n\n{}\n\nNow think like a master post-collapse innovator. Connect distant dots across domains. Invent creative, practical solutions. Extract every actionable, life-saving detail. Build the most sophisticated, clever, and effective DIY response possible.", context)
                } else {
                    "Proxy parsing error".to_string()
                }
            },
            Err(e) => format!("Failed to connect to Rust Proxy: {}", e),
        }
    }

    #[tool(description = "Use this tool when a single search query is not enough, or for extremely complex situations requiring new insights. It will search the database broadly multiple times to find distant connections across multiple domains.")]
    async fn iterative_rag_thinking(&self, #[tool(aggr)] args: IterativeRagThinkingArgs) -> String {
        let payload = ProxySearchPayload {
            query: args.initial_query,
            top_k: args.max_iterations * 3,
            domain_filter: None,
        };
        
        match self.client.post("http://127.0.0.1:8000/search")
            .json(&payload)
            .send()
            .await 
        {
            Ok(resp) => {
                if let Ok(data) = resp.json::<ProxySearchResponse>().await {
                    if data.results.is_empty() {
                        return "Iterative depth search returned no context. Rely on base training.".to_string();
                    }
                    let mut context = String::new();
                    for (i, r) in data.results.iter().enumerate() {
                        context.push_str(&format!("[{}] {}\n\n---\n\n", i + 1, r));
                    }
                    
                    format!("Iterative deep synthesis complete:\n\n{}\n\nSynthesize creatively. Connect unexpected domains. Invent new solutions. Provide the most advanced, clever, and beneficial survival response.", context)
                } else {
                    "Proxy parsing error".to_string()
                }
            },
            Err(e) => format!("Failed to connect to Rust Proxy: {}", e),
        }
    }

    #[tool(description = "Use when analyzing or improving survival-related scripts, automation tools, or DIY projects. Always request diffs for full creative insight.")]
    async fn search_git_history(&self, #[tool(aggr)] args: SearchGitHistoryArgs) -> String {
        let mut cmd = Command::new("git");
        cmd.arg("log").arg(format!("-n {}", args.limit));
        
        if args.include_diffs {
            cmd.arg("-p");
        }
        cmd.current_dir(&args.project_path);
        
        match cmd.output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if output.status.success() {
                    if stdout.trim().is_empty() {
                        "The git repository has no commit history yet.".to_string()
                    } else {
                        format!("Git History for {}:\n\n{}", args.project_path, stdout)
                    }
                } else {
                    format!("Git command failed: {}", String::from_utf8_lossy(&output.stderr))
                }
            },
            Err(e) => format!("Failed to execute git command: {}", e),
        }
    }

    #[tool(description = "Execute arbitrary bash commands on the local Linux terminal. Use this to read files, compile code, run diagnostics, or manage systemd services autonomously.")]
    async fn execute_terminal_command(&self, #[tool(aggr)] args: ExecuteTerminalCommandArgs) -> String {
        use tokio::process::Command as AsyncCommand;
        use tokio::time::timeout;
        use std::time::Duration;
        
        let mut cmd = AsyncCommand::new("bash");
        cmd.arg("-c").arg(&args.command);
        
        let exec = cmd.output();
        match timeout(Duration::from_secs(args.timeout as u64), exec).await {
            Ok(Ok(output)) => {
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                
                let mut resp = format!("Terminal Execution Completed (Exit Code: {})\n", 
                    output.status.code().unwrap_or(-1));
                    
                if !stdout.is_empty() {
                    if stdout.len() > 4000 {
                        stdout = format!("{}\n...[OUTPUT TRUNCATED DUE TO LENGTH LIMIT]", &stdout[..4000]);
                    }
                    resp.push_str(&format!("\n--- STDOUT ---\n{}", stdout));
                }
                if !stderr.is_empty() {
                    resp.push_str(&format!("\n--- STDERR ---\n{}", stderr));
                }
                if stdout.is_empty() && stderr.is_empty() {
                    resp.push_str("\n(Command executed silently with no terminal output)");
                }
                
                resp
            },
            Ok(Err(e)) => format!("Failed to execute terminal command: {}", e),
            Err(_) => format!("Critical Alert: The command timed out after {} seconds and was forcefully aborted.", args.timeout),
        }
    }

    #[tool(description = "
    Direct access to Google Gemini Flash 2.0 for complex reasoning,
    synthesis, and analysis. Uses the full power of Google's latest model.
    Automatically unavailable when offline — system falls back to local models.
    Use this for: complex multi-step reasoning, large context synthesis,
    factual verification against training knowledge, code generation.
    ")]
    async fn ask_gemini(&self, #[tool(aggr)] args: AskGeminiArgs) -> String {
        // Check connectivity via proxy backend status
        match self.client
            .get("http://127.0.0.1:8000/backend")
            .send()
            .await 
        {
            Ok(resp) => {
                if let Ok(status) = resp.json::<serde_json::Value>().await {
                    if !status["online"].as_bool().unwrap_or(false) {
                        return "[Gemini Unavailable] No internet connection detected. \
                                Local LM Studio is active as fallback.".to_string();
                    }
                }
            }
            Err(_) => return "[Gemini Unavailable] Could not reach proxy.".to_string(),
        }

        // Route through proxy's Gemini endpoint
        let payload = serde_json::json!({
            "prompt":      args.prompt,
            "max_tokens":  args.max_tokens,
            "temperature": 0.7,
        });

        match self.client
            .post("http://127.0.0.1:8000/gemini/complete")
            .json(&payload)
            .send()
            .await 
        {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        if let Some(error) = data.get("error") {
                            // Backend detected an offline condition
                            return format!("Gemini Proxy Error: {}", error.as_str().unwrap_or(""));
                        }
                        data["content"].as_str().unwrap_or("Empty response from Gemini").to_string()
                    },
                    Err(e) => format!("Gemini response parse error: {}", e),
                }
            },
            Err(e) => format!("Gemini request failed: {}", e),
        }
    }
}

use rmcp::model::{InitializeResult, Implementation};

#[tool(tool_box)]
impl ServerHandler for OfflineRagMcp {
    fn get_info(&self) -> InitializeResult {
        InitializeResult {
            server_info: Implementation {
                name: "Survival-DIY-MCP".into(),
                version: "0.1.0".into(),
            },
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = OfflineRagMcp::new();
    let transport = (stdin(), stdout());
    
    // serve it via standard input/output
    let server = service.serve(transport).await?;
    
    server.waiting().await?;
    Ok(())
}
