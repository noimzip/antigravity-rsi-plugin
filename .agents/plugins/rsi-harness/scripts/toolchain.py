"""Project toolchain auto-detection and adaptive inner-check resolution."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MAKE_TEST_TARGET_RE = re.compile(r"^test\s*:", re.MULTILINE)


@dataclass
class ToolchainInfo:
    """Detected project language, test runner tool, command, and default timeout."""

    language: str
    tool: str
    check_cmd: str
    default_timeout_sec: int
    detected_files: list[str] = field(default_factory=list)


def _pyproject_uses_pytest(pyproject_path: Path) -> bool:
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    tool = data.get("tool")
    if isinstance(tool, dict) and "pytest" in tool:
        return True
    blob = json.dumps(data).lower()
    return "pytest" in blob


def _detect_python_toolchain(cwd: Path) -> ToolchainInfo | None:
    pyproject = cwd / "pyproject.toml"
    tests_dir = cwd / "tests"
    has_tests_dir = tests_dir.is_dir()
    unittest_cmd = (
        "python3 -m unittest discover -s tests"
        if has_tests_dir
        else "python3 -m unittest discover"
    )

    if pyproject.is_file():
        if _pyproject_uses_pytest(pyproject):
            return ToolchainInfo("python", "pytest", "pytest", 30, ["pyproject.toml"])
        return ToolchainInfo("python", "unittest", unittest_cmd, 15, ["pyproject.toml"])

    for marker in ("pytest.ini", "conftest.py"):
        if (cwd / marker).is_file():
            return ToolchainInfo("python", "pytest", "pytest", 30, [marker])

    setup_cfg = cwd / "setup.cfg"
    if setup_cfg.is_file():
        try:
            text = setup_cfg.read_text(encoding="utf-8", errors="replace")
            if "[tool:pytest]" in text:
                return ToolchainInfo("python", "pytest", "pytest", 30, ["setup.cfg"])
        except OSError:
            pass
        return ToolchainInfo("python", "unittest", unittest_cmd, 15, ["setup.cfg"])

    for marker in ("setup.py", "requirements.txt"):
        if (cwd / marker).is_file():
            return ToolchainInfo("python", "unittest", unittest_cmd, 15, [marker])

    if has_tests_dir and any(tests_dir.glob("test_*.py")):
        return ToolchainInfo("python", "unittest", unittest_cmd, 15, ["tests"])

    if any(cwd.glob("test_*.py")):
        return ToolchainInfo(
            "python", "unittest", "python3 -m unittest discover", 15, ["test_*.py"]
        )

    return None


def _detect_node_toolchain(cwd: Path) -> ToolchainInfo | None:
    pkg_json = cwd / "package.json"
    tsconfig = cwd / "tsconfig.json"
    if pkg_json.is_file():
        pm = "npm"
        detected = ["package.json"]
        for lockfile, manager in (
            ("pnpm-lock.yaml", "pnpm"),
            ("yarn.lock", "yarn"),
            ("bun.lockb", "bun"),
            ("bun.lock", "bun"),
        ):
            if (cwd / lockfile).is_file():
                pm = manager
                detected.append(lockfile)
                break
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        scripts = data.get("scripts") if isinstance(data, dict) else {}
        test_script = scripts.get("test") if isinstance(scripts, dict) else None
        if isinstance(test_script, str) and test_script.strip() and "no test specified" not in test_script:
            lang = "typescript" if tsconfig.is_file() else "javascript"
            return ToolchainInfo(lang, pm, f"{pm} test", 45, detected)
        if tsconfig.is_file():
            return ToolchainInfo("typescript", "tsc", "npx tsc --noEmit", 45, detected + ["tsconfig.json"])
    elif tsconfig.is_file():
        return ToolchainInfo("typescript", "tsc", "npx tsc --noEmit", 45, ["tsconfig.json"])
    return None


def detect_project_toolchain(cwd: Path) -> ToolchainInfo:
    """Detect primary language/test runner command and adaptive timeout for ``cwd``."""
    py_info = _detect_python_toolchain(cwd)
    if py_info is not None:
        return py_info

    node_info = _detect_node_toolchain(cwd)
    if node_info is not None:
        return node_info

    if (cwd / "Cargo.toml").is_file():
        return ToolchainInfo("rust", "cargo", "cargo test", 120, ["Cargo.toml"])

    if (cwd / "go.mod").is_file():
        return ToolchainInfo("go", "go", "go test ./...", 60, ["go.mod"])

    if (cwd / "pom.xml").is_file():
        return ToolchainInfo("java", "maven", "mvn test", 120, ["pom.xml"])

    for gradle_file in ("build.gradle", "build.gradle.kts"):
        if (cwd / gradle_file).is_file():
            cmd = "./gradlew test" if (cwd / "gradlew").is_file() else "gradle test"
            return ToolchainInfo("java", "gradle", cmd, 120, [gradle_file])

    if (cwd / "CMakeLists.txt").is_file():
        return ToolchainInfo("cpp", "ctest", "ctest --output-on-failure", 60, ["CMakeLists.txt"])

    for mk_name in ("Makefile", "makefile"):
        mk_path = cwd / mk_name
        if mk_path.is_file():
            try:
                if _MAKE_TEST_TARGET_RE.search(mk_path.read_text(encoding="utf-8", errors="replace")):
                    return ToolchainInfo("make", "make", "make test", 45, [mk_name])
            except OSError:
                pass

    return ToolchainInfo("unknown", "none", "", 30, [])


def resolve_inner_check(
    cwd: Path,
    config: dict[str, Any],
) -> tuple[str, float, ToolchainInfo]:
    """Resolve effective ``(check_cmd, timeout_sec, toolchain_info)`` with explicit precedence."""
    info = detect_project_toolchain(cwd)
    raw_cmd = config.get("inner_check_cmd")
    raw_timeout = config.get("inner_check_timeout_sec")

    if raw_cmd == "":
        check_cmd = ""
    elif raw_cmd is None or str(raw_cmd).strip().lower() == "auto":
        check_cmd = info.check_cmd
    else:
        check_cmd = str(raw_cmd).strip()

    if (
        isinstance(raw_timeout, (int, float))
        and not isinstance(raw_timeout, bool)
        and raw_timeout > 0
    ):
        timeout_sec = float(raw_timeout)
    else:
        timeout_sec = float(info.default_timeout_sec)

    return check_cmd, timeout_sec, info
