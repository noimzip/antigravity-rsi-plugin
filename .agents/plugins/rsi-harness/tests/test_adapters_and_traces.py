"""Unit and integration tests for M2 runtime adapters, event normalizers, OTel traces, and TraceStore."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PLUGIN_DIR / "scripts"
REPO_ROOT = PLUGIN_DIR.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from adapters.runtime import (  # noqa: E402
    AgentRuntimeAdapter,
    AgyAdapter,
    ClaudeCodeAdapter,
    CodexCliAdapter,
    GeminiCliAdapter,
    PtyWrapperAdapter,
    RuntimeRunResult,
    get_runtime_adapter,
    list_runtime_adapters,
    register_runtime_adapter,
)
from common import load_jsonl, summarize_transcript  # noqa: E402
from evolve_outer import run_evolution_cycle  # noqa: E402
from guard_harness import evaluate_tool_guard  # noqa: E402
from inner_gate import evaluate_stop  # noqa: E402
from run_bench import run_benchmark  # noqa: E402
from trace_collector import record_event  # noqa: E402
from trace_schema import (  # noqa: E402
    AgentTrace,
    export_agent_trace_jsonl,
    runtime_result_to_agent_trace,
    transcript_to_agent_trace,
    validate_agent_trace,
)
from trace_store import TraceStore  # noqa: E402


class RuntimeAdapterTests(unittest.TestCase):
    def test_command_building_and_evolution_env(self) -> None:
        cwd = Path("/tmp/fake-wt")
        cases = [
            ("agy", ["agy", "-p", "hello", "--dangerously-skip-permissions", "--output-format", "json", "--model", "m1"]),
            ("claude", ["claude", "-p", "hello", "--output-format", "json", "--dangerously-skip-permissions", "--model", "m1"]),
            ("codex", ["codex", "exec", "--json", "--full-auto", "--model", "m1", "hello"]),
            ("gemini", ["gemini", "-p", "hello", "--output-format", "json", "--yolo", "-m", "m1"]),
        ]
        for name, expected_cmd in cases:
            adapter = get_runtime_adapter(name)
            cmd = adapter.build_command("hello", cwd, model="m1")
            self.assertEqual(cmd, expected_cmd)
            env = adapter.build_env({}, evolution_mode=True, allowed_root=REPO_ROOT)
            self.assertEqual(env["RSI_EVOLUTION_MODE"], "1")
            self.assertEqual(env["RSI_ALLOWED_ROOT"], str(REPO_ROOT.resolve()))

        pty_adapter = PtyWrapperAdapter({"pty": {"command": ["custom-agent", "--prompt", "{prompt}"]}})
        self.assertEqual(
            pty_adapter.build_command("solve task", cwd),
            ["custom-agent", "--prompt", "solve task"],
        )

    def test_output_parsing_across_adapters(self) -> None:
        agy_res = AgyAdapter().parse_output(
            json.dumps({"usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_tokens": 50}}),
            "",
            0,
            0.5,
            ["agy"],
        )
        self.assertEqual(agy_res.total_tokens, 200)
        self.assertEqual(agy_res.cache_read_tokens, 50)

        claude_res = ClaudeCodeAdapter().parse_output(
            json.dumps(
                {
                    "session_id": "cc-1",
                    "result": "fixed",
                    "total_cost_usd": 0.02,
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 40,
                        "cache_read_input_tokens": 300,
                        "cache_creation_input_tokens": 60,
                    },
                }
            ),
            "",
            0,
            0.7,
            ["claude"],
        )
        self.assertEqual(claude_res.input_tokens, 100)
        self.assertEqual(claude_res.output_tokens, 40)
        self.assertEqual(claude_res.cache_read_tokens, 300)
        self.assertEqual(claude_res.total_tokens, 500)
        self.assertEqual(claude_res.metadata["session_id"], "cc-1")

        # Codex single JSON and JSONL stream
        codex_jsonl = "\n".join(
            [
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 200, "cached_input_tokens": 50, "output_tokens": 80},
                    }
                ),
            ]
        )
        codex_res = CodexCliAdapter().parse_output(codex_jsonl, "", 0, 0.4, ["codex"])
        self.assertEqual(codex_res.input_tokens, 200)
        self.assertEqual(codex_res.cache_read_tokens, 50)
        self.assertEqual(codex_res.output_tokens, 80)
        self.assertEqual(codex_res.total_tokens, 280)

        # Gemini stats.models nested tokens
        gemini_payload = {
            "response": "done",
            "stats": {
                "models": {
                    "gemini-2.5-pro": {
                        "tokens": {"prompt": 400, "candidates": 120, "cached": 100, "total": 520}
                    }
                }
            },
        }
        gem_res = GeminiCliAdapter().parse_output(
            json.dumps(gemini_payload), "", 0, 0.9, ["gemini"]
        )
        self.assertEqual(gem_res.input_tokens, 400)
        self.assertEqual(gem_res.output_tokens, 120)
        self.assertEqual(gem_res.cache_read_tokens, 100)
        self.assertEqual(gem_res.total_tokens, 520)

    def test_pty_wrapper_live_execution_and_trace_emission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            adapter = PtyWrapperAdapter(
                {
                    "pty": {
                        "command": [
                            sys.executable,
                            "-c",
                            "import sys; print('PTY running: ' + sys.argv[1]); print('tokens=432')",
                        ]
                    }
                }
            )
            res = adapter.run("hello-pty", cwd, timeout_sec=10)
            self.assertEqual(res.exit_code, 0)
            self.assertIn("PTY running: hello-pty", res.stdout)
            self.assertEqual(res.total_tokens, 432)
            self.assertEqual(len(res.trace_paths), 2)
            for tp in res.trace_paths:
                self.assertTrue(Path(tp).is_file())

    def test_registry_aliases_custom_registration_and_run_benchmark_claude(self) -> None:
        self.assertIsInstance(get_runtime_adapter("claude-code"), ClaudeCodeAdapter)
        self.assertIsInstance(get_runtime_adapter("codex-cli"), CodexCliAdapter)
        self.assertIsInstance(get_runtime_adapter("gemini-cli"), GeminiCliAdapter)
        with self.assertRaises(ValueError):
            get_runtime_adapter("nonexistent-runtime")

        class _DummyAdapter(AgentRuntimeAdapter):
            name = "dummy-test"

            def build_command(self, prompt, cwd, *, model=None, extra_args=None):
                return ["echo", prompt]

            def parse_output(self, stdout, stderr, exit_code, duration_sec, command):
                return RuntimeRunResult(self.name, command, exit_code, stdout=stdout)

        register_runtime_adapter("dummy-test", _DummyAdapter)
        self.assertIn("dummy-test", list_runtime_adapters())
        self.assertIsInstance(get_runtime_adapter("dummy-test"), _DummyAdapter)

        # Fake `claude` binary invoked via run_benchmark(..., runner="claude")
        bin_dir = Path(tempfile.mkdtemp(prefix="fake-claude-bin-"))
        self.addCleanup(lambda: shutil.rmtree(bin_dir, ignore_errors=True))
        claude_bin = bin_dir / "claude"
        claude_bin.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "from pathlib import Path\n"
            "cwd = Path.cwd()\n"
            "sol = cwd / 'solution.py'\n"
            "txt = sol.read_text(encoding='utf-8') if sol.is_file() else ''\n"
            "if 'safe_ratio_sum' in txt:\n"
            "    sol.write_text('def safe_ratio_sum(pairs):\\n    return sum(a/b for a,b in pairs) if pairs else 0.0\\n')\n"
            "print(json.dumps({'result': 'ok', 'usage': {'input_tokens': 300, 'output_tokens': 100}}))\n",
            encoding="utf-8",
        )
        claude_bin.chmod(claude_bin.stat().st_mode | stat.S_IEXEC)
        orig_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
        self.addCleanup(lambda: os.environ.__setitem__("PATH", orig_path))

        report = run_benchmark(REPO_ROOT, REPO_ROOT / "benchmarks" / "tasks.json", runner="claude")
        self.assertEqual(report["runner"], "claude")
        self.assertEqual(report["total_tasks"], 3)
        self.assertEqual(report["total_tokens"], 1200)


class EventNormalizerTests(unittest.TestCase):
    def test_guard_across_antigravity_claude_codex_and_gemini_payloads(self) -> None:
        os.environ.pop("RSI_EVOLUTION_MODE", None)

        # Antigravity blocked
        ag_block = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {"name": "write_file", "args": {"Path": ".agents/plugins/rsi-harness/hooks.json"}},
            }
        )
        self.assertEqual(ag_block["decision"], "ask")

        # Claude Code Write blocked -> hookSpecificOutput format
        cc_block = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "session_id": "cc-sess-1",
                "tool_name": "Write",
                "tool_input": {"file_path": ".agents/plugins/rsi-harness/hooks.json"},
            }
        )
        self.assertEqual(cc_block["hookSpecificOutput"]["hookEventName"], "PreToolUse")
        self.assertEqual(cc_block["hookSpecificOutput"]["permissionDecision"], "ask")

        # Claude Code Bash redirect blocked
        cc_bash_block = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "session_id": "cc-sess-1",
                "tool_name": "Bash",
                "tool_input": {"command": "echo x > benchmarks/tasks.json"},
            }
        )
        self.assertEqual(cc_bash_block["hookSpecificOutput"]["permissionDecision"], "ask")

        # Claude Code allowed edit
        cc_allow = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "session_id": "cc-sess-1",
                "tool_name": "Edit",
                "tool_input": {"file_path": "src/tasklib.py"},
            }
        )
        self.assertEqual(cc_allow["hookSpecificOutput"]["permissionDecision"], "allow")

        # Codex blocked
        codex_block = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "call_id": "cx-1",
                "name": "shell",
                "arguments": json.dumps({"command": "rm benchmarks/tasks.json"}),
            }
        )
        self.assertEqual(codex_block["decision"], "ask")

        # Gemini blocked & allowed
        gem_block = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "functionCall": {"name": "write_file", "args": {"file_path": "benchmarks/tasks.json"}},
            }
        )
        self.assertEqual(gem_block["decision"], "ask")
        gem_allow = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "functionCall": {"name": "read_file", "args": {"command": "cat benchmarks/tasks.json"}},
            }
        )
        self.assertEqual(gem_allow["decision"], "allow")

    def test_trace_collector_and_inner_gate_runtime_agnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / "rsi.config.json").write_text(
                json.dumps(
                    {
                        "inner_check_cmd": "python3 -c \"import sys; print('ValueError: fail'); sys.exit(1)\"",
                        "max_inner_retries": 2,
                    }
                ),
                encoding="utf-8",
            )
            transcript = cwd / "cc_transcript.jsonl"
            transcript.write_text(
                json.dumps(
                    {
                        "step_index": 0,
                        "status": "DONE",
                        "input_tokens": 50,
                        "cache_read_tokens": 10,
                        "output_tokens": 40,
                        "tool_calls": [{"name": "Edit"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = record_event(
                {
                    "session_id": "cc-conv-1",
                    "cwd": str(cwd),
                    "transcript_path": str(transcript),
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "src/tasklib.py"},
                    "hook_event_name": "PostToolUse",
                },
                "post_tool",
            )
            self.assertEqual(summary["conversationId"], "cc-conv-1")
            self.assertEqual(summary["runtime"], "claude_code")
            self.assertEqual(summary["total_tokens"], 100)
            events = load_jsonl(cwd / ".rsi" / "traces" / "cc-conv-1" / "events.jsonl")
            self.assertEqual(events[0]["toolName"], "Edit")
            self.assertEqual(events[0]["runtime"], "claude_code")

            # Claude Code stop payload -> returns decision: "block"
            cc_stop = evaluate_stop(
                {
                    "session_id": "cc-conv-1",
                    "cwd": str(cwd),
                    "transcript_path": str(transcript),
                    "stop_hook_active": False,
                }
            )
            self.assertEqual(cc_stop["decision"], "block")
            self.assertIn("ValueError: fail", cc_stop["reason"])

            # stop_hook_active=True -> allows stop immediately
            cc_active = evaluate_stop(
                {
                    "session_id": "cc-conv-1",
                    "cwd": str(cwd),
                    "transcript_path": str(transcript),
                    "stop_hook_active": True,
                }
            )
            self.assertEqual(cc_active, {})

            # Antigravity stop payload -> returns decision: "continue"
            agy_stop = evaluate_stop(
                {
                    "conversationId": "agy-conv-1",
                    "cwd": str(cwd),
                    "continuationActive": False,
                }
            )
            self.assertEqual(agy_stop["decision"], "continue")


class AgentTraceSchemaTests(unittest.TestCase):
    def test_transcript_to_agent_trace_opt_in_content_and_mcp_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            transcript = cwd / "transcript.jsonl"
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "step_index": 0,
                                "type": "USER_INPUT",
                                "status": "DONE",
                                "content": "secret prompt text",
                                "input_tokens": 100,
                                "cache_read_tokens": 25,
                                "output_tokens": 0,
                                "latency_ms": 120.0,
                            }
                        ),
                        json.dumps(
                            {
                                "step_index": 1,
                                "type": "PLANNER_RESPONSE",
                                "status": "DONE",
                                "content": "assistant completion text",
                                "input_tokens": 20,
                                "cache_read_tokens": 0,
                                "output_tokens": 45,
                                "latency_ms": 230.0,
                                "tool_calls": [
                                    {"name": "read_file", "args": {"Path": "a.py"}},
                                    {"name": "mcp_github_list_issues", "args": {"owner": "x"}},
                                ],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            # Default include_content=False omits prompt/completion
            t_redacted = transcript_to_agent_trace(
                transcript,
                trace_id="trace-1",
                runtime="antigravity",
                model="test-model",
                cwd=cwd,
                include_content=False,
            )
            d_redacted = t_redacted.to_dict()
            ok, errs = validate_agent_trace(d_redacted)
            self.assertTrue(ok, msg=str(errs))
            self.assertEqual(d_redacted["token_usage"]["total_tokens"], 190)
            self.assertEqual(d_redacted["otel_attributes"]["gen_ai.usage.input_tokens"], 120)
            self.assertEqual(d_redacted["otel_attributes"]["gen_ai.usage.output_tokens"], 45)
            self.assertEqual(d_redacted["latency_ms"], 350.0)
            kinds = [s["kind"] for s in d_redacted["execution_steps"]]
            self.assertEqual(kinds, ["llm", "llm", "tool", "mcp"])
            for s in d_redacted["execution_steps"]:
                self.assertNotIn("gen_ai.prompt", s["attributes"])
                self.assertNotIn("gen_ai.completion", s["attributes"])

            # Opt-in include_content=True captures prompt/completion & tool args
            t_full = transcript_to_agent_trace(
                transcript,
                trace_id="trace-2",
                runtime="antigravity",
                cwd=cwd,
                include_content=True,
            )
            d_full = t_full.to_dict()
            self.assertEqual(
                d_full["execution_steps"][0]["attributes"]["gen_ai.prompt"],
                "secret prompt text",
            )
            self.assertEqual(
                d_full["execution_steps"][1]["attributes"]["gen_ai.completion"],
                "assistant completion text",
            )

            # Exported agent_trace.jsonl is readable by summarize_transcript
            exported = export_agent_trace_jsonl(t_redacted, cwd / "agent_trace.jsonl", overwrite=True)
            sum_legacy = summarize_transcript(transcript)
            sum_otel = summarize_transcript(exported)
            self.assertEqual(sum_otel["total_tokens"], sum_legacy["total_tokens"])
            self.assertEqual(sum_otel["tool_calls"], 2)
            self.assertEqual(sum_otel["latency_ms"], 350.0)

    def test_validate_agent_trace_rejects_invalid_and_converts_runtime_result(self) -> None:
        res = RuntimeRunResult(
            runtime_name="claude",
            command=["claude", "-p", "hi"],
            exit_code=0,
            input_tokens=80,
            output_tokens=20,
            cache_read_tokens=10,
            total_tokens=110,
            duration_sec=0.45,
        )
        trace = runtime_result_to_agent_trace(
            res,
            trace_id="rt-1",
            parent_harness_version="sha-parent",
            candidate_harness_version="sha-cand",
            cwd="/tmp/repo",
        )
        ok, errs = validate_agent_trace(trace.to_dict())
        self.assertTrue(ok, msg=str(errs))
        roundtrip = AgentTrace.from_dict(trace.to_dict())
        self.assertEqual(roundtrip.otel_attributes["gen_ai.system"], "anthropic")
        self.assertEqual(roundtrip.latency_ms, 450.0)

        # Validation failures
        bad_missing = trace.to_dict()
        del bad_missing["trace_id"]
        ok_m, errs_m = validate_agent_trace(bad_missing)
        self.assertFalse(ok_m)
        self.assertTrue(any("trace_id" in e for e in errs_m))

        bad_neg = trace.to_dict()
        bad_neg["token_usage"]["input_tokens"] = -5
        ok_n, errs_n = validate_agent_trace(bad_neg)
        self.assertFalse(ok_n)
        self.assertIn("invalid_token_field:input_tokens", errs_n)

        bad_otel = trace.to_dict()
        del bad_otel["otel_attributes"]["gen_ai.system"]
        ok_o, errs_o = validate_agent_trace(bad_otel)
        self.assertFalse(ok_o)
        self.assertIn("missing_otel_attr:gen_ai.system", errs_o)

    def test_trace_collector_emits_validated_agent_trace_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            record_event(
                {
                    "conversationId": "conv-otel",
                    "cwd": str(cwd),
                    "stepIdx": 1,
                    "toolCall": {"name": "mcp_context7_query-docs", "args": {"libraryId": "x"}},
                },
                "post_tool",
            )
            at_path = cwd / ".rsi" / "traces" / "conv-otel" / "agent_trace.jsonl"
            self.assertTrue(at_path.is_file())
            records = load_jsonl(at_path)
            self.assertEqual(len(records), 1)
            ok, errs = validate_agent_trace(records[0])
            self.assertTrue(ok, msg=str(errs))
            self.assertEqual(records[0]["execution_steps"][0]["kind"], "mcp")


class TraceStoreTests(unittest.TestCase):
    def _create_temp_git_repo(self, dest: Path) -> None:
        shutil.copytree(
            REPO_ROOT,
            dest,
            ignore=shutil.ignore_patterns(".git", ".rsi", "__pycache__"),
            dirs_exist_ok=True,
        )
        subprocess.run(["git", "init", "-b", "main"], cwd=dest, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "RSI Test"], cwd=dest, check=True)
        subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=dest, check=True)
        subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=dest, check=True, capture_output=True)

    def test_store_records_multiple_generations_queries_greps_and_mounts_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            self._create_temp_git_repo(repo)

            r_leak = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock", mock_bad_proposal="leak")
            r_budget = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock", mock_bad_proposal="budget")
            r_ok = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock")

            self.assertEqual(r_leak["store_generation"], 1)
            self.assertEqual(r_budget["store_generation"], 2)
            self.assertEqual(r_ok["store_generation"], 3)

            store = TraceStore(repo)
            self.assertEqual(len(store.list_generations()), 3)
            self.assertEqual(len(store.list_generations(accepted=True)), 1)
            self.assertEqual(len(store.list_generations(accepted=False)), 2)

            # Verify uncompressed artifacts persisted per generation
            gen3_dir = Path(r_ok["store_dir"])
            self.assertTrue((gen3_dir / "proposal.diff").is_file())
            self.assertTrue((gen3_dir / "config_snapshot.json").is_file())
            self.assertTrue((gen3_dir / "metadata.json").is_file())
            self.assertTrue((gen3_dir / "harness_files" / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md").is_file())
            self.assertTrue((gen3_dir / "traces" / "mock-task_01_edge_case" / "summary.json").is_file())
            self.assertTrue((gen3_dir / "candidate_traces" / "mock-task_03_multi_file_rule" / "summary.json").is_file())

            # Query filters
            by_gen = store.query(generation=2)
            self.assertEqual(len(by_gen), 1)
            self.assertEqual(by_gen[0]["run_id"], r_budget["run_id"])

            by_task = store.query(task_id="task_03_multi_file_rule")
            self.assertEqual(len(by_task), 3)

            by_err = store.query(error_type="critic_benchmark_leakage")
            self.assertEqual(len(by_err), 1)
            self.assertEqual(by_err[0]["generation"], 1)

            by_text = store.query(text="padding rule line 42")
            self.assertEqual(len(by_text), 1)
            self.assertEqual(by_text[0]["generation"], 2)

            # Regex grep across uncompressed traces and diffs
            hits = store.grep_traces(r"parse_kv_lines", generation=1)
            self.assertTrue(any(h["file"] == "proposal.diff" for h in hits))

            bundle = store.get_generation_bundle(3)
            self.assertIsNotNone(bundle)
            self.assertIn("AGENTS.md", "\n".join(bundle["harness_files"]))

            # Read-only worktree mount uses symlinks instead of copying store
            wt = Path(tmp) / "wt"
            wt.mkdir()
            mounted = store.mount_readonly_into_worktree(wt)
            self.assertIn("store", mounted)
            self.assertTrue((wt / ".rsi" / "store").is_symlink())
            self.assertTrue((wt / ".rsi" / "traces_archive").is_symlink())
            self.assertTrue((wt / ".rsi" / "evolution_history.jsonl").is_symlink())
            self.assertFalse((wt / ".rsi" / "traces").is_symlink())

            # CLI query & grep subcommands work
            q_out = subprocess.check_output(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "trace_store.py"),
                    "--repo",
                    str(repo),
                    "query",
                    "--accepted",
                    "true",
                ],
                text=True,
            )
            q_parsed = json.loads(q_out)
            self.assertEqual(len(q_parsed), 1)
            self.assertEqual(q_parsed[0]["generation"], 3)

            g_out = subprocess.check_output(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "trace_store.py"),
                    "--repo",
                    str(repo),
                    "grep",
                    "--pattern",
                    "padding rule line 10",
                ],
                text=True,
            )
            g_parsed = json.loads(g_out)
            self.assertGreaterEqual(len(g_parsed), 1)


if __name__ == "__main__":
    unittest.main()
