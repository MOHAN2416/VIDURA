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
    from developer import CodeChangeGenerator, CodeChangeApplier
    from tools.developer import ProposeCodeChangeTool, ApplyCodeChangeTool

    permission_manager = PermissionManager(default_write_allowed=False)
    code_applier = CodeChangeApplier(permission_manager=permission_manager, workspace_root=config.workspace_root)
    code_generator = CodeChangeGenerator(workspace_root=config.workspace_root, model=provider, codebase_manager=codebase_manager, applier=code_applier)

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

    # Display Startup Banner
    caps = [k for k, v in provider.capabilities.to_dict().items() if v]
    print("=" * 60)
    print(f"{config.app_name}")
    print(f"Provider: {provider.provider_name}")
    print(f"Model: {provider.model_name}")
    if hasattr(router, "developer_provider_type"):
        print(f"Developer Provider: {router.developer_provider_type} ({router.developer_model_name})")
    print(f"Capabilities: {', '.join(caps)}")
    print(f"Database: {config.db_path}")
    print(f"Codebase Index: {summary['python_files']} Python files | {summary['total_modules']} modules")
    print(f"Registered Tools: {', '.join(t.name for t in tool_registry.list_tools())}")
    print("=" * 60)
    print("Commands: /remember, /recall, /memories, /forget, /codebase, /modules, /symbol, /dependencies, /importers, /approve, /deny, /exit\n")
    
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
                    permission_manager.grant_write_permission()
                    pending = code_applier.get_pending_proposal()
                    if pending:
                        pending.status = "approved"
                        print(f"✅ Write permission EXPLICITLY GRANTED for pending proposal on '{pending.target_file}'.\n")
                    else:
                        print("✅ Write permission EXPLICITLY GRANTED for this session.\n")
                    continue

                if user_input.lower() == "/deny":
                    permission_manager.revoke_write_permission()
                    code_applier.clear_pending_proposal()
                    print("🛑 Write permission REVOKED and active pending proposal cleared.\n")
                    continue

                # Handle explicit memory commands
                if handle_memory_command(user_input, memory_manager):
                    continue

                # Handle explicit codebase commands
                if handle_codebase_command(user_input, codebase_manager):
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


def main() -> None:
    """VIDURA Main Entry Point."""
    run_cli()


if __name__ == "__main__":
    main()
