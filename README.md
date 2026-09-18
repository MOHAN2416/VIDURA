# VIDURA: Final Release (Phase 14)

> **Autonomous Capability within Absolute Permission Boundaries.**  
> VIDURA is a permission-controlled, local-first AI software engineering assistant and controlled self-development system implemented in Python.

---

## 1. System Overview & Mission

VIDURA is designed to bridge the gap between high-capability code generation models and deterministic software engineering discipline. Rather than granting language models arbitrary shell or filesystem access, VIDURA wraps model intelligence in strict application-level guardrails:

- **Local-First AI**: Operates offline by default using Ollama and high-efficiency local models (`gemma4:e4b-it-qat` / `qwen2.5:3b-instruct`).
- **Permission-Controlled Code Modification**: Code generation produces read-only *proposals*. No file is written, created, or modified without explicit user authorization (`/approve`).
- **Deterministic Codebase Intelligence**: Uses Python's native Abstract Syntax Tree (`ast`) and hybrid RAG indexing rather than LLM speculation to understand code structure.
- **Post-Change Physical Verification**: Every applied change is verified against the filesystem byte-for-byte and validated with syntax checks and automated test runs.
- **Empirical Experience Memory**: Retains structured outcomes of past developer actions and tests to avoid repeating past failures.

---

## 2. Core Architectural Principles

1. **Local-First Execution**: The system is fully operational without Internet access or external APIs.
2. **Deterministic Precedence over LLM Output**: The filesystem, AST parser, and test suite are authoritative facts. Model natural language output is never accepted as proof of file creation, modification, deletion, or test success.
3. **Fail-Closed Security**: Any ambiguity, boundary breach, sensitive data match, or lack of permission immediately halts operations and denies execution.
4. **Single-Cycle Developer & Self-Development Boundaries**: One user request or self-development goal corresponds to exactly one cycle. Unbounded recursion, self-re-invocation, and continuous auto-retry loops are prohibited.
5. **No Model Weight Fine-Tuning**: Learning occurs purely through structured experience retrieval, lesson extraction, and dynamic prompt context assembly.

---

## 3. Complete Phase Roadmap

| Phase | Subsystem / Capability | Status |
| :--- | :--- | :--- |
| **Phase 1** | Project Foundation, Configuration, Logging | ✅ Completed |
| **Phase 2** | Local Offline LLM Provider (`qwen2.5:3b-instruct` / `gemma4:e4b-it-qat`) | ✅ Completed |
| **Phase 3** | Deterministic Agent Loop & JSON Decision Protocol | ✅ Completed |
| **Phase 4** | Safe Read-Only Filesystem Tools & Workspace Boundary Validation | ✅ Completed |
| **Phase 5** | Persistent Memory Subsystem (`SQLite` backend, 4 categories, search) | ✅ Completed |
| **Phase 6** | Codebase Intelligence (AST parser, symbol tables, dependency graph) | ✅ Completed |
| **Phase 7** | Permission-Controlled Code Modification (Proposals, Verifier, Rollbacks) | ✅ Completed |
| **Phase 8** | Developer Agent (Plan -> Propose -> Approve -> Apply -> Verify -> Test) | ✅ Completed |
| **Phase 9** | Cloud Model Routing, Fallback, Usage Limits & Sensitive Data Protection | ✅ Completed |
| **Phase 10** | Controlled Self-Development Loop (Meta-analysis, security-critical gates) | ✅ Completed |
| **Phase 11** | Experience Memory & Learning (Failure recording, empirical lessons) | ✅ Completed |
| **Phase 12** | RAG + Advanced Codebase Intelligence (Dense embeddings, hybrid retrieval) | ✅ Completed |
| **Phase 13** | **Multi-Agent System** | ❌ **CANCELLED** |
| **Phase 14** | **Final Integration, Hardening, Validation & Release Preparation** | 🚀 **FINAL RELEASE** |

> [!IMPORTANT]
> **Roadmap Conclusion**: Phase 13 (*Multi-Agent System*) has been explicitly CANCELLED. VIDURA does not use multi-agent communication networks or autonomous peer subagents. **Phase 14 is the final architectural milestone of VIDURA.**

---

## 4. Local Model Configuration & Hardware Requirements

VIDURA is optimized for resource-constrained hardware and operates comfortably under CPU-only environments.

- **Primary Local Model**: `gemma4:e4b-it-qat` (Ollama)
- **Alternative Local Model**: `qwen2.5:3b-instruct` (Ollama)
- **Resource Constraints**:
  - **RAM**: Strict execution envelope under **16 GB RAM** (Idle footprint < 50 MB RSS; peak heap < 100 MB).
  - **Compute**: CPU-only execution supported (AVX2 acceleration recommended via Ollama).
  - **Storage**: < 50 MB for core dependencies; dynamic SQLite storage (`vidura.db` ~ 6.3 MB).
- **Timeout & Failure Safety**: Local provider calls have deterministic timeouts and fail gracefully to structured user-facing errors if the Ollama daemon is unreachable.

---

## 5. Cloud Model Configuration, Routing & Security

VIDURA provides an optional, application-controlled cloud extension via `OllamaCloudProvider`:

- **Cloud Model**: `gemma4:31b-cloud`
- **Application-Controlled 5-Tier Routing Hierarchy**:
  1. *Tier 1: Security & Offline Policies*: Targets with secrets (`.env`, `*.key`) or marked `local_only` force Local execution.
  2. *Tier 2: Global Configuration Check*: If `VIDURA_CLOUD_ENABLED=false`, requests are locked locally.
  3. *Tier 3: Explicit Mode Enforcement*: `VIDURA_ROUTING_MODE=local` forces local; `cloud` checks usage limits.
  4. *Tier 4: Configured Role Overrides*: Developer agent routed according to `VIDURA_DEVELOPER_MODEL_PROVIDER`.
  5. *Tier 5: Dynamic Task Complexity*: Simple/Moderate tasks use Local; Complex/Critical tasks route to Cloud.
- **Bounded Fallback**: Transient network or server errors (HTTP 502/503/504) automatically fall back to the local provider.
- **Permanent Errors Fail Closed**: Cloud security violations, credential detections, and configuration errors **never** silently fall back; they fail with explicit error codes.
- **Credential Isolation & Redaction**: All cloud-bound prompts pass through deterministic regex filters redacting API keys, authorization headers, private keys, and environment files.

---

## 6. Agent Loop & Tool Execution

The core agent loop (`agent/loop.py`) runs a deterministic state machine:

1. Formulates strict system instructions injecting available tools, anti-speculation rules, and anti-fabrication mandates.
2. Accepts structured JSON actions:
   - `{"action": "respond", "content": "..."}`
   - `{"action": "think", "content": "..."}`
   - `{"action": "tool_call", "tool_name": "...", "arguments": {...}}`
3. Enforces a maximum step bound (`max_steps=10`) to prevent runaway execution.
4. Suppresses all hallucinated natural language claims regarding file modification, file deletion, and test execution.

---

## 7. Deterministic Codebase Intelligence

Located in `codebase/`, this subsystem performs machine-level analysis of Python source trees:

- **AST Parsing**: Inspects module docstrings, class definitions, function signatures, return annotations, line spans, and import statements.
- **Dependency & Importer Graphs**: Builds directional graphs tracking which modules import or are imported by target components.
- **Strictly Read-Only**: Analyzes files in memory without modifying the filesystem.
- **Exclusion Boundaries**: Automatically excludes `.git`, `.venv`, `__pycache__`, build directories, and files matching sensitive filename patterns.

---

## 8. Developer Agent & Controlled Code Modification

VIDURA's developer pipeline (`developer/`) executes code modification strictly through an auditable, human-in-the-loop lifecycle:

```
[User Request]
       │
       ▼
[Task Understanding & Classification]
       │
       ▼
[Developer Planning & Dependency Analysis]
       │
       ▼
[Code Change Proposal Generation (READ-ONLY)]
       │
       ▼
[Human Review Boundary: /approve or /deny]
       │
 ┌─────┴────────────────────────┐
 │ (Approved)                   │ (Denied)
 ▼                              ▼
[Filesystem Write (Atomic)]    [Proposal Discarded]
 │                              [Write Permission Revoked]
 ▼
[Post-Change Physical Verification (Byte-for-Byte)]
 │
 ▼
[Automated Test Execution (pytest)]
```

### Unsupported Operations Policy
- File deletion, directory removal, renaming, moving, unlinking, and arbitrary shell command execution are **strictly unsupported**.
- Only `create_file` and `modify_file` within the workspace boundary are permitted.
- Any attempt by an LLM to claim a file was deleted is intercepted and suppressed with an explicit warning.

---

## 9. Persistent Memory Subsystem

VIDURA manages persistent state across CLI sessions in `data/vidura.db` using SQLite:

- **Categories**:
  - `user`: User preferences, naming styles, formatting requirements.
  - `project`: Architecture conventions, framework constraints.
  - `conversation`: Summarized past discussions.
  - `experience`: Empirical developer outcomes and post-change test logs.
- **Deduplication**: SHA-256 content hashing prevents duplicate entries while updating hit counts.
- **Full-Text Retrieval**: Keyword and semantic search recall relevant memories into the prompt context.

---

## 10. Controlled Self-Development Loop

VIDURA can improve its own codebase (`self_development/`) under rigid safety constraints:

- **Explicit Feature Gate**: Must be enabled via `VIDURA_SELF_DEVELOPMENT_ENABLED=true`.
- **Protected Security Infrastructure**: The loop cannot propose or apply modifications to security-critical files:
  - `permissions/manager.py`
  - `developer/applier.py`
  - `models/security.py`
  - `config.py`
  - Any file outside the workspace.
- **Recursion Guard**: Self-development cycles cannot invoke self-development cycles (`SelfDevelopmentRecursionError`).
- **Approval Gate**: Proposes changes and halts. Self-development can never grant write permissions to itself.

---

## 11. Experience Memory & Empirical Learning

Phase 11 introduces learning from past developer experiences without modifying model weights:

- **Failure & Success Capture**: When a developer task or test fails, VIDURA records the affected files, failure reason, and actionable lesson into SQLite.
- **Scored Retrieval**: When future tasks target the same files or modules, relevant past experiences are retrieved and scored.
- **Warning Injection**: Lessons from past failures are injected into the developer prompt as explicit negative constraints (e.g., *"Warning: Previous change to auth/jwt.py failed because key length was < 2048 bits"*).

---

## 12. RAG & Advanced Codebase Intelligence

Phase 12 enhances codebase understanding through a local hybrid retrieval pipeline:

- **Code-Aware Chunking**: Chunks source files by AST function and class boundaries rather than arbitrary token splits.
- **Dense Vector Embeddings**: Uses local Ollama embeddings (`nomic-embed-text` / `all-minilm`) with automatic hash-based incremental indexing.
- **Hybrid Multi-Signal Retrieval**: Combines:
  1. Dense semantic cosine similarity
  2. Exact symbol & lexical matches
  3. AST import dependency graph relationships
  4. Past experience memory lessons
- **Graceful Degradation**: If vector models or the RAG database are unavailable, VIDURA falls back seamlessly to deterministic AST indexing without interrupting the agent.

---

## 13. Comprehensive CLI Command Reference

| Command | Category | Description |
| :--- | :--- | :--- |
| `/help` | Core | Displays full command manual and operational guidelines. |
| `/exit`, `/quit` | Core | Safely terminates the interactive session and closes databases. |
| `/approve` | Developer | Explicitly authorizes application of the pending code proposal. |
| `/deny` | Developer | Explicitly rejects pending proposal and revokes write access. |
| `/pending` | Developer | Displays detailed diff and rationale of active pending proposal. |
| `/test [target]` | Testing | Executes verified automated test suite or specific test file. |
| `/codebase` | Intelligence | Displays workspace overview, file metrics, and symbol counts. |
| `/modules` | Intelligence | Lists all scanned Python modules and class definitions. |
| `/symbol <name>` | Intelligence | Deterministically finds functions, classes, or methods. |
| `/dependencies <mod>` | Intelligence | Lists local modules imported by target module. |
| `/importers <mod>` | Intelligence | Lists local modules that import target module. |
| `/rag status` | RAG | Reports RAG indexing status, chunk count, and embedding dimensions. |
| `/rag index` | RAG | Triggers incremental scan and vector indexing of workspace source files. |
| `/rag query <query>` | RAG | Executes hybrid search and displays retrieved ranked code chunks. |
| `/rag clear` | RAG | Clears the vector index and chunk database. |
| `/remember <text>` | Memory | Stores explicit fact or preference into persistent SQLite memory. |
| `/recall <query>` | Memory | Searches stored memories by keyword query. |
| `/memories` | Memory | Lists all stored memories across all categories. |
| `/forget <id>` | Memory | Deletes a stored memory record by ID. |

---

## 14. Security & Safety Model (21-Point Regression Matrix)

VIDURA maintains a 21-point security matrix verified on every build:

1. **Path Traversal Prevention**: Absolute and relative escapes (`../../etc/passwd`) are denied.
2. **Absolute Path Escape**: Paths outside `workspace_root` fail boundary validation.
3. **Symlink Escape Prevention**: Target symlinks pointing outside the workspace are blocked via `os.path.realpath`.
4. **Protected File Protection**: Writes to `.git`, `.env`, `*.key`, `*.pem`, and `*.secret` are rejected.
5. **Secret File Exclusion**: Sensitive files are omitted from AST indexing and RAG chunking.
6. **Default No-Write Policy**: Write permission is denied by default until explicitly granted by the user.
7. **Single-Use Permission**: Authorizing a change revokes write permission immediately after completion.
8. **Stale Proposal Rejection**: Proposals cannot be applied if the underlying file on disk was modified.
9. **Atomic Syntax Validation**: Malformed Python syntax proposals are rejected prior to filesystem modification.
10. **Explicit Denial Handling**: `/deny` revokes write access and discards the pending proposal cleanly.
11. **Tool Argument Validation**: Extraneous or missing tool parameters are rejected deterministically.
12. **Cloud Credential Isolation**: Sensitive API tokens and keys are redacted from cloud-bound prompts.
13. **Cloud Data Leakage Prevention**: Prompts containing private key headers are blocked from transmission.
14. **RAG Secret Redaction**: Chunker redacts credentials before calculating vector embeddings.
15. **RAG Stale Hash Detection**: Incremental index uses SHA-256 hashes to prevent stale context injection.
16. **Experience Memory Deduplication**: Duplicate experiences update evidence counts rather than polluting memory.
17. **Security Infrastructure Immunity**: Self-development loop cannot touch permissions, security, or applier code.
18. **Self-Development Recursion Guard**: Active cycles cannot invoke nested cycles.
19. **Anti-Fabrication (File Modification)**: Natural language claims of file modification are suppressed if no change occurred.
20. **Anti-Fabrication (File Deletion)**: Deletion claims are suppressed and reported as strictly unsupported.
21. **Anti-Fabrication (Test Execution)**: Claims of test success are suppressed unless verified by `pytest` exit codes.

---

## 15. Installation & Setup Guide

### Prerequisites
- Linux OS (Ubuntu 22.04+ or similar)
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally
- `uv` package manager (recommended) or standard `venv`

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/MOHAN2416/VIDURA.git
cd VIDURA-main
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 2: Download Local Ollama Models
```bash
ollama pull gemma4:e4b-it-qat
ollama pull nomic-embed-text
```

### Step 3: Configure Environment Variables
Copy example configuration:
```bash
cp .env.example .env
```

Review or customize core `.env` settings:
```ini
# Local Model
VIDURA_MODEL_PROVIDER=local
VIDURA_LOCAL_MODEL=gemma4:e4b-it-qat
VIDURA_OLLAMA_HOST=http://localhost:11434

# Routing & Cloud
VIDURA_ROUTING_MODE=auto
VIDURA_CLOUD_ENABLED=false
VIDURA_CLOUD_FALLBACK_ENABLED=true

# Developer & Self-Development
VIDURA_DEVELOPER_MODEL_PROVIDER=local
VIDURA_SELF_DEVELOPMENT_ENABLED=false
VIDURA_AUTO_APPLY_CHANGES=false
```

---

## 16. Quick Start & Example Workflows

### Starting VIDURA
```bash
.venv/bin/python3 main.py
```

### Example 1: Codebase Inspection
```text
vidura> /codebase
[INFO] Scanned 45 modules, 128 classes, 312 functions.

vidura> /symbol AgentLoop
[INFO] AgentLoop defined in agent/loop.py:43 (class)
```

### Example 2: Developer Task with User Approval
```text
vidura> Add a helper function to calculate factorial in utils/math.py
[INFO] Generated Plan & Code Change Proposal (prop_9b4a12c8)
Target: utils/math.py (create_file)
Type /pending to review diff.
Type /approve to apply or /deny to cancel.

vidura> /approve
[INFO] Writing file: utils/math.py
[INFO] Post-change verification succeeded: utils/math.py content verified on disk.
[INFO] Tests passed: 1 passed in 0.05s.
```

---

## 17. Testing & Verification Guide

VIDURA includes a rigorous test suite of **456 automated unit, integration, and security tests**.

### Run Complete Regression Suite
```bash
.venv/bin/pytest -v
```

### Run Phase 14 Security & Hardening Suite
```bash
.venv/bin/pytest tests/test_phase14.py -vv
```

### Run Specific Subsystem Test Suites
- Agent loop tests: `.venv/bin/pytest tests/test_agent.py`
- Codebase intelligence: `.venv/bin/pytest tests/test_codebase.py`
- Developer pipeline: `.venv/bin/pytest tests/test_developer*.py`
- Model router & fallback: `.venv/bin/pytest tests/test_model_router.py tests/test_cloud_*.py`
- Experience memory: `.venv/bin/pytest tests/test_experience.py`
- RAG & Chunking: `.venv/bin/pytest tests/test_rag.py`

---

## 18. Configuration Options Reference

| Environment Variable | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `VIDURA_MODEL_PROVIDER` | `local` | `local`, `cloud` | Default provider for general conversation |
| `VIDURA_LOCAL_MODEL` | `gemma4:e4b-it-qat` | Local Ollama tags | Active local LLM identifier |
| `VIDURA_CLOUD_MODEL` | `gemma4:31b-cloud` | Cloud Ollama tags | Active cloud LLM identifier |
| `VIDURA_ROUTING_MODE` | `auto` | `auto`, `local`, `cloud` | Deterministic router operational mode |
| `VIDURA_CLOUD_ENABLED` | `false` | `true`, `false` | Master toggle for cloud model access |
| `VIDURA_CLOUD_FALLBACK_ENABLED` | `true` | `true`, `false` | Fallback to local on transient cloud errors |
| `VIDURA_DEVELOPER_MODEL_PROVIDER`| `local` | `local`, `cloud` | Model provider for developer agent code generation |
| `VIDURA_SELF_DEVELOPMENT_ENABLED`| `false` | `true`, `false` | Master gate for self-development cycles |
| `VIDURA_WORKSPACE_ROOT` | Current dir | Valid directory path | Security boundary for all file operations |
| `VIDURA_DB_PATH` | `data/vidura.db` | File path | SQLite persistent database location |

---

## 19. Architectural Non-Goals & Exclusions

To preserve stability, verifiability, and security, VIDURA explicitly excludes:

1. **Multi-Agent Systems (Phase 13 CANCELLED)**: No multi-agent negotiation, distributed autonomous peer agents, or subagent message buses.
2. **Model Weight Training**: No local fine-tuning, LoRA adapters, or backpropagation.
3. **Unrestricted Shell Execution**: No arbitrary `bash` or `subprocess.Popen` execution by LLM decision.
4. **Automated Continuous Self-Improvement**: Self-development is single-cycle and strictly halted at the user permission boundary.
5. **File Deletion / Destruction**: Deletions are fundamentally unsupported by design.

---

## 20. Project Status & Conclusion

**PHASE 14 COMPLETE: VIDURA IS IN FINAL RELEASE STATUS.**

- All 13 completed phases (1 through 12, and 14) are integrated, verified, and hardened.
- Zero open regressions across 456 automated test scenarios.
- Deterministic guardrails enforce safe code evolution with human oversight at every step.
