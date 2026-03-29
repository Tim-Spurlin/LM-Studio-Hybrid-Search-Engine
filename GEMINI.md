# Global Antigravity Rules: Multi-Agent Synchronization

When working in this environment, especially if multiple agents or subagents are dispatched, you MUST explicitly adhere to the following coordination and synchronization protocols:

1. **State Awareness & Traceability**
   - Before taking any action or modifying a file, always check the relevant `task.md` or centralized planning artifacts (e.g., `implementation_plan.md` in the `.gemini` folder or workspace root) to understand the current global state of the project.
   - Always document your intended actions and completed steps in `task.md` (or the relevant tracker) so other agents can see exactly what has been done and what is currently in progress.

2. **Strict Task Segregation (No Overlap)**
   - Never simultaneously edit the same file or work on the exact same checklist item as another agent.
   - If you see a task currently marked as in-progress (`[/]`) by another agent in `task.md`, do not touch it. Pick a different, distinctly unassigned task (`[ ]`).
   - Break large tasks down into mutually exclusive, component-level tasks so parallel work is naturally separated.

3. **Explicit Handoffs & Dependencies**
   - If your task depends on another agent's output, explicitly verify their task is marked complete (`[x]`) before proceeding. 
   - Leave clear, concise notes in your artifacts (like `walkthrough.md`) explaining your design decisions. This ensures other agents can easily follow your logic without repeating your analysis.

4. **Resource Conflict Avoidance**
   - When modifying shared dependencies, architecture, or configuration files (such as global CSS, routing files, or database schemas), complete these tasks synchronously before fanning out to parallel component tasks.
   - Use `grep_search` to verify you aren't duplicating variables, functions, or UI elements that another agent might have simply renamed.

# Global Workspace Rules: Project "River Talk" / OfflineRAG

## 1. Context Awareness Protocol
Before writing code or debugging, acknowledge that this workspace operates a "Hub and Spoke" architecture:
- **Hub:** Python Proxy (`proxy_server.py`) on Port `8000`.
- **Spokes:** Android App, Desktop App, LM Studio, Browser Extension, Daemons.
- **Primary Model:** `liquid/lfm2.5-1.2b` (running on LM Studio, Port 8888) is the **sole active advanced reasoning model** for the entire ecosystem. It handles all native multi-step reasoning, hypothesis synthesis, and context inference.
- **Never** suggest connecting Spokes directly to LM Studio (Port 8888); they must always route through the Proxy for RAG context.

## 2. Network Topology Truths
- **Local LLM Port:** 8888 (Strictly Localhost).
- **Proxy Port:** 8000 (Bound to Tailscale IP `100.114.240.44`).
- **Remote Access:** Exclusively via Tailscale VPN. Do not suggest Cloudflare tunnels, Ngrok, or port forwarding.
- **Security:** Always use Bearer Token `sk-lm-48IXAjXt:SpEuhK6SYL8sSI0YKiVb` for API calls.

## 3. Environment Specifics (Arch Linux)
- **Display Server:** Wayland. Do not suggest `xclip` or `x11` specific solutions. Use `wl-clipboard` (`wl-paste`).
- **Kernel:** Zen Kernel. Optimize for throughput/low latency.
- **File Paths:** Reference `~/Desktop/Projects/Local LLM` for core logic.

## 4. Code Modification Rules
- **Python Proxy / IMDDS Preservation:** When updating `proxy_server.py`, you **MUST preserve** the LangGraph state machine, the Hybrid Search logic, and the `IMDDS_TruthFilter`. Never remove, bypass, or alter the biomedical hypothesis generation or the `filtered_scored = imdds.filter_and_score(retrieved_chunks)` logic. Doing so breaks the entire vector database connection.
- **Rust Daemon:** Hardcoded file routing rules exist for a reason (to prevent AI hallucinations). Do not re-introduce AI-based file sorting for `~/Downloads` unless explicitly asked.
- **Android:** Maintain `network_security_config.xml` allowing cleartext traffic to `100.114.240.44`.

## 5. Memory & Database
- **Database:** The system uses HDF5 (`universal_knowledge_base.h5`), NOT SQL or Pinecone.
- **Ingestion:** Files go to `~/KnowledgeDrop`. Do not manually edit the HDF5 file.

## 6. Safety Checks
- If a proposed change affects the Proxy Port or IP, warn that it will break the Android App connection.
- If the user reports "missing downloads," check the `intelligent-organizer` daemon logs first.
