import sys
import logging
from config import load_config
from models import ModelRouter, ProviderConfigurationError
from tools import (
    ToolRegistry,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    InspectCodebaseTool,
    FindSymbolTool,
    FindImportersTool,
    FindDependenciesTool,
)
from memory import MemoryStore, MemoryManager, MemoryCategory
from codebase import CodebaseManager
from agent import Agent

logger = logging.getLogger("VIDURA")


def handle_memory_command(cmd_input: str, memory_manager: MemoryManager) -> bool:
    """Handles explicit CLI memory commands (/remember, /recall, /memories, /forget)."""
    parts = cmd_input.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/remember":
        if not arg:
            print("Usage: /remember <fact or preference to save>\n")
            return True
        mem = memory_manager.remember(content=arg, memory_type=MemoryCategory.USER)
        print(f"Memory saved. [ID: {mem.id}]\n")
        return True

    elif cmd == "/recall":
        results = memory_manager.recall(query=arg)
        if not results:
            print("No matching memories found.\n")
        else:
            print("\n--- Recalled Memories ---")
            for m in results:
                print(f"• [{m.id[:8]}] ({m.type}): {m.content}")
            print()
        return True

    elif cmd == "/memories":
        results = memory_manager.list_memories(limit=50)
        if not results:
            print("No stored memories.\n")
        else:
            print("\n--- Stored Memories ---")
            for m in results:
                print(f"• ID: {m.id}\n  Category: {m.type}\n  Content: {m.content}\n  Created: {m.created_at[:19]}\n")
        return True

    elif cmd == "/forget":
        if not arg:
            print("Usage: /forget <memory_id>\n")
            return True
        deleted = memory_manager.forget(memory_id=arg)
        if deleted:
            print(f"Memory '{arg}' deleted.\n")
        else:
            print(f"Memory '{arg}' not found.\n")
        return True

    return False


def handle_experience_command(cmd_input: str, experience_manager: Any) -> bool:
    """Handles explicit CLI experience and learning commands (/experiences, /experience, /lessons, /experience-search)."""
    parts = cmd_input.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/experiences":
        records = experience_manager.list_experiences(limit=20)
        if not records:
            print("No stored experiences.\n")
        else:
            print(f"\n--- Stored Experiences ({len(records)}) ---")
            for r in records:
                status_icon = "✅" if r.success else "❌"
                f_type = f" [{r.failure_type}]" if r.failure_type else ""
                print(f"{status_icon} ID: {r.id[:8]} | Task: {r.task[:50]}")
                print(f"   Outcome: {'SUCCESS' if r.success else 'FAILURE'}{f_type}")
                if r.lesson:
                    print(f"   Lesson: {r.lesson[:100]}")
                print()
        return True

    elif cmd == "/experience":
        if not arg:
            print("Usage: /experience <experience_id>\n")
            return True
        exp = experience_manager.get_experience(arg)
        if not exp:
            all_exps = experience_manager.list_experiences(limit=100)
            matches = [e for e in all_exps if e.id.startswith(arg)]
            exp = matches[0] if matches else None
        if not exp:
            print(f"Experience '{arg}' not found.\n")
        else:
            status_str = "SUCCESS" if exp.success else "FAILURE"
            print("\n--- Experience Record Details ---")
            print(f"ID:           {exp.id}")
            print(f"Task:         {exp.task}")
            print(f"Type:         {exp.task_type}")
            print(f"Outcome:      {status_str}")
            if exp.failure_type:
                print(f"Failure Type: {exp.failure_type}")
            if exp.failure_reason:
                print(f"Failure Reason: {exp.failure_reason}")
            if exp.affected_files:
                print(f"Files:        {', '.join(exp.affected_files)}")
            if exp.relevant_symbols:
                print(f"Symbols:      {', '.join(exp.relevant_symbols)}")
            if exp.provider or exp.model:
                print(f"Model:        {exp.model} ({exp.provider})")
            print(f"Result:       {exp.result}")
            print(f"Lesson:       {exp.lesson}")
            if exp.recommendation:
                print(f"Caution/Rec:  {exp.recommendation}")
            print(f"Confidence:   {exp.confidence:.2f}")
            print(f"Created:      {exp.created_at[:19]}")
            print()
        return True

    elif cmd == "/lessons":
        lessons = experience_manager.derive_lessons(component=arg if arg else None)
        if not lessons:
            print("No lessons derived from current experiences.\n")
        else:
            print(f"\n--- Derived Empirical Lessons ({len(lessons)}) ---")
            for lsn in lessons:
                print(f"• Topic: {lsn.topic} (Confidence: {lsn.confidence:.2f}, {lsn.success_count} passed, {lsn.failure_count} failed)")
                print(f"  Lesson: {lsn.lesson}")
                if lsn.recommendation:
                    print(f"  Recommendation: {lsn.recommendation}")
                print()
        return True

    elif cmd == "/experience-search":
        if not arg:
            print("Usage: /experience-search <query>\n")
            return True
        from experience.models import ExperienceQuery
        q = ExperienceQuery(query=arg, limit=10, min_score=0.1)
        scored = experience_manager.retrieve_experiences(q)
        if not scored:
            print(f"No experiences matching query '{arg}'.\n")
        else:
            print(f"\n--- Experiences matching '{arg}' ({len(scored)}) ---")
            for s in scored:
                status_icon = "✅" if s.experience.success else "⚠️"
                print(f"{status_icon} [{s.score:.2f}] {s.experience.task[:60]}")
                if s.experience.lesson:
                    print(f"   Lesson: {s.experience.lesson[:100]}")
                print(f"   Breakdown: {s.score_breakdown}")
                print()
        return True

    return False


def handle_codebase_command(cmd_input: str, codebase_manager: CodebaseManager) -> bool:
    """Handles explicit CLI codebase intelligence commands (/codebase, /modules, /symbol, /dependencies, /importers)."""
    parts = cmd_input.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/codebase":
        summary = codebase_manager.get_summary()
        print("\n--- Codebase Summary ---")
        print(f"Total Files: {summary['total_files']} (Python Source Files: {summary['python_files']})")
        print(f"Modules: {summary['total_modules']}")
        print(f"Classes: {summary['total_classes']}")
        print(f"Functions/Methods: {summary['total_functions']}")
        print(f"Top-level Packages: {', '.join(summary['top_level_packages'])}")
        if summary["parse_error_count"] > 0:
            print(f"⚠️  Parse Errors: {summary['parse_error_count']} files")
        print()
        return True

    elif cmd == "/modules":
        modules = codebase_manager.list_modules()
        print(f"\n--- Scanned Modules ({len(modules)}) ---")
        for m in modules:
            cls_str = f" [Classes: {', '.join(c.name for c in m.classes)}]" if m.classes else ""
            print(f"• {m.name} ({m.relative_path}){cls_str}")
        print()
        return True

    elif cmd == "/symbol":
        if not arg:
            print("Usage: /symbol <class_or_function_name>\n")
            return True
        symbols = codebase_manager.find_symbol(arg)
        if not symbols:
            print(f"No symbol matching '{arg}' found in codebase index.\n")
        else:
            print(f"\n--- Symbols matching '{arg}' ({len(symbols)}) ---")
            for s in symbols:
                kind = s.get("kind", "symbol").upper()
                name = s.get("name", "")
                path = s.get("file_path", "")
                line = s.get("line_number", 1)
                doc = f" - \"{s['docstring'].splitlines()[0]}\"" if s.get("docstring") else ""
                print(f"• [{kind}] {name} -> {path}:L{line}{doc}")
            print()
        return True

    elif cmd == "/dependencies":
        if not arg:
            print("Usage: /dependencies <module_name>\n")
            return True
        deps = codebase_manager.find_dependencies(arg)
        print(f"\n--- Dependencies of '{arg}' ({len(deps)}) ---")
        if not deps:
            print("No local module dependencies found.")
        else:
            for d in deps:
                print(f"• {d}")
        print()
        return True

    elif cmd == "/importers":
        if not arg:
            print("Usage: /importers <module_name>\n")
            return True
        importers = codebase_manager.find_importers(arg)
        print(f"\n--- Importers of '{arg}' ({len(importers)}) ---")
        if not importers:
            print("No local modules import this module.")
        else:
            for imp in importers:
                print(f"• {imp}")
        print()
        return True

    return False


def handle_rag_command(cmd_input: str, rag_manager: Any) -> bool:
    """Handles explicit CLI RAG commands (/rag status, /rag index, /rag rebuild, /rag search, /rag context)."""
    if not rag_manager:
        return False
    parts = cmd_input.strip().split(maxsplit=2)
    if not parts:
        return False
    raw_cmd = parts[0].lower()

    # Normalize "/rag <subcmd>" vs "/rag-<subcmd>"
    subcmd = ""
    arg = ""
    if raw_cmd == "/rag" and len(parts) > 1:
        subcmd = parts[1].lower()
        arg = parts[2].strip() if len(parts) > 2 else ""
    elif raw_cmd.startswith("/rag-"):
        subcmd = raw_cmd[5:]
        arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
    elif raw_cmd == "/rag":
        subcmd = "status"
    else:
        return False

    if subcmd in ("search", "find"):
        if not arg:
            print("Usage: /rag search <query>\n")
            return True
        from rag.models import RAGQuery
        results = rag_manager.retrieve(RAGQuery(query=arg, top_k=5))
        if not results:
            print(f"No RAG results found for '{arg}'.\n")
        else:
            print(f"\n--- RAG Search Results for '{arg}' ({len(results)} chunks) ---")
            for i, r in enumerate(results, 1):
                c = r.chunk
                loc = f"{c.file_path}:{c.start_line}-{c.end_line}"
                name_str = f" ({c.name})" if c.name else ""
                print(f"#{i} [{c.chunk_type.upper()}] {loc}{name_str}")
                print(f"   Score: {r.score:.3f} (Vector: {r.similarity_score:.3f}, Lexical: {r.lexical_score:.3f})")
                snippet = c.content.strip().splitlines()
                first_lines = "\n   ".join(snippet[:4])
                print(f"   Snippet:\n   {first_lines}")
                if len(snippet) > 4:
                    print(f"   ... ({len(snippet) - 4} more lines)")
                print()
        return True

    elif subcmd in ("context", "ctx"):
        if not arg:
            print("Usage: /rag context <task_description>\n")
            return True
        from rag.models import RAGQuery
        ctx = rag_manager.assemble_context(RAGQuery(query=arg, top_k=5))
        print("\n--- Assembled Codebase Context ---")
        print(ctx.prompt_text)
        print(f"\nTotal Chunks: {ctx.total_chunks} | Estimated Tokens: {ctx.total_tokens}\n")
        return True

    elif subcmd in ("index", "scan"):
        print("\nIndexing codebase for RAG...")
        stats = rag_manager.index_codebase()
        print("RAG Indexing complete:")
        print(f"• Total Chunks: {stats.get('total_chunks', 0)}")
        print(f"• Files Indexed: {stats.get('indexed_files', 0)}")
        if stats.get("errors"):
            print(f"⚠️  Errors: {len(stats['errors'])} files encountered errors")
        print()
        return True

    elif subcmd in ("rebuild", "refresh"):
        print("\nRebuilding RAG index from scratch...")
        rag_manager.clear()
        stats = rag_manager.index_codebase(force=True)
        print("RAG Index Rebuild complete:")
        print(f"• Total Chunks: {stats.get('total_chunks', 0)}")
        print(f"• Files Indexed: {stats.get('indexed_files', 0)}")
        print()
        return True

    elif subcmd in ("status", "stats"):
        status = rag_manager.get_status()
        print("\n--- RAG Status & Statistics ---")
        print(f"RAG Enabled:     {'Yes' if status.get('enabled') else 'No'}")
        print(f"Total Chunks:    {status.get('total_chunks', 0)}")
        print(f"Indexed Files:   {status.get('indexed_files', 0)}")
        print(f"Embedding Model: {status.get('embedding_model')}")
        print(f"Dimensions:      {status.get('dimensions')}")
        print(f"Storage:         {status.get('storage_type')}")
        print(f"Database:        {status.get('db_path')}")
        print()
        return True

    return False



def handle_usage_command(cmd_input: str, router: ModelRouter) -> bool:
    """Handles /usage CLI command to display cloud session usage and limits."""
    cmd = cmd_input.strip().lower()
    if cmd == "/usage":
        tracker = getattr(router, "usage_tracker", None)
        if not tracker:
            print("Usage tracking is not available.\n")
            return True
        summary = tracker.get_summary()
        print("\n--- VIDURA Cloud Usage & Limits ---")
        print(f"Cloud Status: {'enabled' if getattr(router, 'cloud_enabled', True) else 'disabled'}")
        print(f"Cloud Requests Attempted: {summary['cloud_requests_attempted']}")
        print(f"Cloud Requests Successful: {summary['cloud_requests_successful']}")
        print(f"Cloud Requests Failed: {summary['cloud_requests_failed']}")
        print(f"Fallback Invocations: {summary['fallback_requests']}")
        print(f"Developer Cloud Requests: {summary['developer_cloud_requests']}")
        print(f"Local Model Requests: {summary['local_requests']}")

        limit_req = summary['max_requests'] if summary['max_requests'] is not None else "Unlimited"
        limit_dev = summary['max_developer_requests'] if summary['max_developer_requests'] is not None else "Unlimited"
        print(f"Request Limit: {summary['cloud_requests_attempted']} / {limit_req}")
        print(f"Developer Cloud Limit: {summary['developer_cloud_requests']} / {limit_dev}")

        in_tok = summary['input_tokens'] if summary['input_tokens'] is not None else 0
        out_tok = summary['output_tokens'] if summary['output_tokens'] is not None else 0
        tot_tok = summary['total_tokens'] if summary['total_tokens'] is not None else 0
        print(f"Tokens (est): {in_tok} in / {out_tok} out / {tot_tok} total")

        if summary['cost_status'] == "estimated_configured":
            print(f"Estimated Cost: ${summary['estimated_cost']:.6f} (configured pricing)")
        else:
            print("Estimated Cost: not_configured (pricing not set)")
        print()
        return True
    return False


def handle_self_development_command(cmd_input: str, sd_loop: Any) -> bool:
    """Handles /selfdev and /self-dev CLI commands to initiate a self-development cycle."""
    cmd_raw = cmd_input.strip()
    lower = cmd_raw.lower()
    if lower.startswith("/selfdev") or lower.startswith("/self-dev"):
        parts = cmd_raw.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print("Usage: /selfdev <goal_description>\nExample: /selfdev Improve VIDURA's malformed tool-call handling.\n")
            return True
        goal_text = parts[1].strip()

        from self_development import SelfDevelopmentGoal, SelfDevelopmentDisabledError

        try:
            goal = SelfDevelopmentGoal(description=goal_text)
            print(f"\nInitiating self-development cycle for goal: '{goal_text}'...")
            result = sd_loop.initiate_cycle(goal)
            print(f"\n{result.summary}\n")
        except SelfDevelopmentDisabledError as err:
            print(f"\n❌ {err}\n")
        except Exception as err:
            logger.error(f"Self-development error: {err}", exc_info=True)
            print(f"\n❌ Self-development error: {err}\n")
        return True
    return False


def run_cli() -> None:
    """Runs the VIDURA CLI interactive chat loop using the Agent system."""
    config = load_config()
    
    logger.info("Initializing VIDURA Model Router...")
    try:
        router = ModelRouter(config=config)
        provider = router
    except ProviderConfigurationError as err:
        print(f"❌ Configuration Error: {err}")
        sys.exit(1)
    
    # Initialize Persistent Memory Store and Manager
    memory_store = MemoryStore(db_path=config.db_path)
    memory_manager = MemoryManager(store=memory_store)
    logger.info(f"Initialized MemoryManager with SQLite at '{config.db_path}'.")

    # Initialize Codebase Manager and scan project workspace
    codebase_manager = CodebaseManager(workspace_root=config.workspace_root)
    summary = codebase_manager.scan()
    logger.info(
        f"Initialized CodebaseManager: Scanned {summary['python_files']} Python files ({summary['total_modules']} modules)."
    )

    # Initialize Permission Manager and Developer Code Change System
    from permissions import PermissionManager
    from developer import CodeChangeGenerator, CodeChangeApplier, DeveloperExecutor
    from tools.developer import ProposeCodeChangeTool, ApplyCodeChangeTool
    from experience.manager import ExperienceManager

    experience_manager = getattr(memory_manager, "experience_manager", None) or ExperienceManager(store=memory_store)
    permission_manager = PermissionManager(default_write_allowed=False)
    code_applier = CodeChangeApplier(permission_manager=permission_manager, workspace_root=config.workspace_root)
    code_generator = CodeChangeGenerator(workspace_root=config.workspace_root, model=provider, codebase_manager=codebase_manager, applier=code_applier)
    developer_executor = DeveloperExecutor(
        applier=code_applier,
        permission_manager=permission_manager,
        workspace_root=config.workspace_root,
        codebase_manager=codebase_manager,
        experience_manager=experience_manager,
    )

    # Initialize Phase 10 Self-Development Loop
    from self_development import SelfDevelopmentLoop
    sd_loop = SelfDevelopmentLoop(
        config=config,
        model=provider,
        codebase_manager=codebase_manager,
        memory_manager=memory_manager,
        permission_manager=permission_manager,
        generator=code_generator,
        executor=developer_executor,
        workspace_root=config.workspace_root,
    )

    # Initialize Phase 12 RAG Manager
    from rag.manager import RAGManager
    rag_manager = RAGManager(
        workspace_root=config.workspace_root,
        db_path=config.db_path,
        codebase_manager=codebase_manager,
        experience_manager=experience_manager,
    )

    # Initialize Tool Registry and register read-only and permission-controlled tools
    tool_registry = ToolRegistry()
    tool_registry.register(ListDirectoryTool())
    tool_registry.register(ReadFileTool())
    tool_registry.register(SearchFilesTool())
    tool_registry.register(InspectCodebaseTool(codebase_manager))
    tool_registry.register(FindSymbolTool(codebase_manager))
    tool_registry.register(FindImportersTool(codebase_manager))
    tool_registry.register(FindDependenciesTool(codebase_manager))
    tool_registry.register(ProposeCodeChangeTool(code_generator))
    tool_registry.register(ApplyCodeChangeTool(code_applier))

    # Register Developer Testing Tool (Phase 8/14)
    from tools.developer import RunTestsTool
    tool_registry.register(RunTestsTool(permission_manager=permission_manager, workspace_root=config.workspace_root))

    # Register Phase 12 RAG Tools
    from tools.rag import SearchCodebaseSemanticTool, GetRelevantCodeContextTool
    tool_registry.register(SearchCodebaseSemanticTool(rag_manager))
    tool_registry.register(GetRelevantCodeContextTool(rag_manager))

    # Display Startup Banner
    caps = [k for k, v in provider.capabilities.to_dict().items() if v]
    print("=" * 60)
    print(f"{config.app_name} (Phase 14 Final Release)")
    print(f"Provider: {provider.provider_name}")
    print(f"Model: {provider.model_name}")
    if hasattr(router, "cloud_enabled"):
        print(f"Cloud Status: {'enabled' if router.cloud_enabled else 'disabled'}")
    if hasattr(router, "routing_mode"):
        print(f"Routing Mode: {router.routing_mode}")
    if hasattr(router, "developer_provider_type"):
        print(f"Developer Provider: {router.developer_provider_type} ({router.developer_model_name})")
    if hasattr(router, "fallback_enabled"):
        print(f"Cloud Fallback: {'enabled' if router.fallback_enabled else 'disabled'}")
    print(f"Self-Development: {'enabled' if config.vidura_self_development_enabled else 'disabled'}")
    print(f"Experience Memory: {'enabled' if getattr(config, 'vidura_experience_memory_enabled', True) else 'disabled'}")
    print(f"RAG Code Intelligence: {'enabled' if getattr(config, 'vidura_rag_enabled', True) else 'disabled'} ({getattr(config, 'vidura_rag_embedding_model', 'deterministic')})")
    print(f"Capabilities: {', '.join(caps)}")
    print(f"Database: {config.db_path}")
    print(f"Codebase Index: {summary['python_files']} Python files | {summary['total_modules']} modules")
    print(f"Registered Tools: {', '.join(t.name for t in tool_registry.list_tools())}")
    print("=" * 60)
    print("Commands: /help, /remember, /recall, /memories, /forget, /experiences, /experience, /lessons, /rag search, /rag status, /codebase, /symbol, /usage, /selfdev, /approve, /deny, /exit\n")

    
    # Check connection and model availability
    connected, status_msg = provider.check_connection()
    if not connected:
        print(f"⚠️  Connection Warning: {status_msg}")
        if getattr(router, "provider_type", "") == "cloud":
            print("To configure Ollama Cloud, export OLLAMA_API_KEY=<your_key> or VIDURA_CLOUD_API_KEY=<your_key>.\n")
        else:
            print("Please check that Ollama is running locally and the model is pulled.\n")

    # Initialize Agent orchestrator
    agent = Agent(
        model=provider,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        codebase_manager=codebase_manager,
        max_steps=config.agent_max_steps,
        rag_manager=rag_manager,
    )

    try:
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue

                if user_input.lower() in ("/exit", "/quit"):
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "/approve":
                    if sd_loop.active_cycle and sd_loop.active_cycle.status == "pending_permission":
                        print("Applying approved self-development proposal...")
                        res = sd_loop.apply_approved(explicit_permission=True)
                        print(f"\n{res.summary}\n")
                        continue

                    permission_manager.grant_write_permission()
                    pending = code_applier.get_pending_proposal()
                    if pending:
                        pending.status = "approved"
                        print(f"✅ Write permission EXPLICITLY GRANTED for pending proposal on '{pending.target_file}'.\n")
                    else:
                        print("✅ Write permission EXPLICITLY GRANTED for this session.\n")
                    continue

                if user_input.lower() == "/deny":
                    if sd_loop.active_cycle and sd_loop.active_cycle.status == "pending_permission":
                        res = sd_loop.deny_active_proposal()
                        print(f"\n{res.summary}\n")
                        continue

                    permission_manager.revoke_write_permission()
                    code_applier.clear_pending_proposal()
                    print("🛑 Write permission REVOKED and active pending proposal cleared.\n")
                    continue

                if user_input.lower() in ("/help", "help"):
                    print("\n--- VIDURA CLI Commands ---")
                    print("  /remember <text>       - Store a user or project memory")
                    print("  /recall [query]        - Search or recall memories")
                    print("  /memories              - List recent stored memories")
                    print("  /forget <id>           - Delete a specific memory")
                    print("  /experiences           - List developer experiences")
                    print("  /experience <id>       - View detailed experience record")
                    print("  /lessons               - View consolidated lessons learned")
                    print("  /experience-search <q> - Search empirical experiences")
                    print("  /rag search <query>    - Hybrid semantic & lexical search")
                    print("  /rag context <task>    - Assemble RAG codebase prompt context")
                    print("  /rag index             - Incrementally index the codebase")
                    print("  /rag rebuild           - Rebuild RAG index from scratch")
                    print("  /rag status            - Display RAG index statistics")
                    print("  /codebase              - Codebase metrics summary")
                    print("  /modules               - List all scanned Python modules")
                    print("  /symbol <name>         - Find classes or functions by name")
                    print("  /dependencies <mod>    - List dependencies of a module")
                    print("  /importers <mod>       - List modules importing a module")
                    print("  /usage                 - Cloud request, token, and cost metrics")
                    print("  /selfdev <goal>        - Run controlled self-development cycle")
                    print("  /approve               - Explicitly grant write permission for pending proposal")
                    print("  /deny                  - Revoke write permission and discard proposal")
                    print("  /help                  - Show this help message")
                    print("  /exit, /quit           - Exit VIDURA\n")
                    continue

                # Handle explicit memory commands
                if handle_memory_command(user_input, memory_manager):
                    continue

                # Handle explicit experience and learning commands
                if handle_experience_command(user_input, experience_manager):
                    continue

                # Handle explicit codebase commands
                if handle_codebase_command(user_input, codebase_manager):
                    continue

                # Handle explicit RAG commands
                if handle_rag_command(user_input, rag_manager):
                    continue

                # Handle usage diagnostics command
                if handle_usage_command(user_input, router):
                    continue

                # Handle self-development command
                if handle_self_development_command(user_input, sd_loop):
                    continue

                try:
                    # Delegate request execution to the Agent orchestrator & loop
                    response_text = agent.run(user_input)
                    print(f"\nVIDURA: {response_text}\n")

                except Exception as err:
                    logger.error(f"Agent Execution Error: {err}", exc_info=True)
                    print(f"\n[Error]: {err}\n")

            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated by user. Goodbye!")
                break
            except Exception as err:
                logger.error(f"Unexpected CLI error: {err}", exc_info=True)
                print(f"\n[Unexpected Error]: {err}\n")
    finally:
        memory_store.close()
        if hasattr(rag_manager, "close"):
            try:
                rag_manager.close()
            except Exception:
                pass


def main() -> None:
    """VIDURA Main Entry Point."""
    run_cli()


if __name__ == "__main__":
    main()
