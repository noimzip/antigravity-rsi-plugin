"""Isolated sandbox adapters (Git worktree, hardened Docker, and remote MicroVM hook)."""

from __future__ import annotations

import importlib
import os
import shutil
import stat
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SandboxConfig:
    """Configuration for isolated sandbox execution backends."""

    backend: str = "worktree"
    docker_image: str = "python:3.12-slim"
    docker_bin: str = "docker"
    network_disabled: bool = True
    cpus: float = 1.0
    memory_mb: int = 1024
    pids_limit: int = 128
    read_only_root: bool = False
    extra_flags: list[str] = field(default_factory=list)
    remote_driver: str | None = None
    remote_endpoint: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SandboxConfig":
        raw = dict(data or {})
        extra = raw.get("extra_flags")
        return cls(
            backend=str(raw.get("backend") or "worktree").strip().lower(),
            docker_image=str(raw.get("docker_image") or "python:3.12-slim"),
            docker_bin=str(raw.get("docker_bin") or "docker"),
            network_disabled=bool(raw.get("network_disabled", True)),
            cpus=float(raw.get("cpus", 1.0)),
            memory_mb=int(raw.get("memory_mb", 1024)),
            pids_limit=int(raw.get("pids_limit", 128)),
            read_only_root=bool(raw.get("read_only_root", False)),
            extra_flags=[str(f) for f in extra] if isinstance(extra, list) else [],
            remote_driver=str(raw["remote_driver"]) if raw.get("remote_driver") else None,
            remote_endpoint=str(raw["remote_endpoint"]) if raw.get("remote_endpoint") else None,
        )


@dataclass
class SandboxExecResult:
    """Result returned by ``SandboxAdapter.exec``."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    command: list[str] = field(default_factory=list)
    timed_out: bool = False


class SandboxAdapter(ABC):
    """Abstract isolation boundary for harness evolution and benchmark grading."""

    backend: str = "base"

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig(backend=self.backend)
        self.workspace: Path | None = None

    @abstractmethod
    def prepare(self, repo: Path, base_sha: str, run_id: str) -> Path:
        """Prepare an isolated workspace from ``repo`` at ``base_sha``."""

    @abstractmethod
    def sync_files(self, src: Path, rel_dest: str, *, read_only: bool = False) -> Path:
        """Copy ``src`` into ``workspace/rel_dest``, optionally enforcing read-only permissions."""

    @abstractmethod
    def exec(
        self,
        cmd: list[str] | str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        readonly_mounts: list[tuple[Path, str]] | None = None,
    ) -> SandboxExecResult:
        """Execute ``cmd`` inside the sandbox."""

    @abstractmethod
    def cleanup(self) -> None:
        """Tear down workspace and temporary sandbox resources."""


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        try:
            os.chmod(dp, dp.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        except OSError:
            pass
        for name in filenames:
            fp = dp / name
            if fp.is_symlink():
                continue
            try:
                os.chmod(fp, fp.stat().st_mode | stat.S_IWUSR)
            except OSError:
                pass


def _apply_readonly(target: Path) -> None:
    if target.is_file() and not target.is_symlink():
        os.chmod(target, 0o444)
    elif target.is_dir():
        for p in target.rglob("*"):
            if p.is_file() and not p.is_symlink():
                os.chmod(p, 0o444)


class GitWorktreeSandbox(SandboxAdapter):
    """Git worktree-backed sandbox isolating candidate edits in ``/tmp``."""

    backend = "worktree"

    def __init__(self, config: SandboxConfig | None = None) -> None:
        super().__init__(config or SandboxConfig(backend="worktree"))
        self.repo: Path | None = None
        self.cand_branch: str | None = None
        self.tmp_root: Path | None = None

    def prepare(self, repo: Path, base_sha: str, run_id: str) -> Path:
        self.repo = repo.resolve()
        self.cand_branch = f"rsi/cand-{run_id}"
        self.tmp_root = Path(tempfile.mkdtemp(prefix="rsi-wt-"))
        self.workspace = self.tmp_root / f"cand-{run_id}"
        subprocess.run(
            ["git", "worktree", "add", "-b", self.cand_branch, str(self.workspace), base_sha],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
            check=True,
        )
        return self.workspace

    def sync_files(self, src: Path, rel_dest: str, *, read_only: bool = False) -> Path:
        if self.workspace is None:
            raise RuntimeError("Sandbox workspace is not prepared")
        rel = Path(rel_dest)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe relative destination inside sandbox: {rel_dest}")
        dest = (self.workspace / rel).resolve(strict=False)
        if not dest.is_relative_to(self.workspace.resolve()):
            raise ValueError(f"Destination escapes sandbox workspace: {rel_dest}")

        src_resolved = src.resolve()
        if src_resolved.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_resolved, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_resolved, dest)
        if read_only:
            _apply_readonly(dest)
        return dest

    def exec(
        self,
        cmd: list[str] | str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        readonly_mounts: list[tuple[Path, str]] | None = None,
    ) -> SandboxExecResult:
        run_cwd = str(cwd or self.workspace or Path.cwd())
        shell = isinstance(cmd, str)
        cmd_list = [cmd] if isinstance(cmd, str) else [str(c) for c in cmd]
        try:
            proc = subprocess.run(
                cmd if shell else cmd_list,
                cwd=run_cwd,
                env=env,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return SandboxExecResult(
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                command=cmd_list,
            )
        except subprocess.TimeoutExpired as exc:
            out_s = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err_s = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return SandboxExecResult(
                returncode=124,
                stdout=out_s,
                stderr=err_s,
                command=cmd_list,
                timed_out=True,
            )

    def cleanup(self) -> None:
        if self.workspace is not None:
            _make_tree_writable(self.workspace)
        if self.repo is not None and self.workspace is not None:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(self.workspace)],
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                check=False,
            )
        if self.repo is not None and self.cand_branch:
            subprocess.run(
                ["git", "branch", "-D", self.cand_branch],
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                check=False,
            )
        if self.tmp_root is not None:
            _make_tree_writable(self.tmp_root)
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.workspace = None


class DockerSandbox(SandboxAdapter):
    """Containerized sandbox running commands inside hardened ``docker run`` isolation."""

    backend = "docker"

    def __init__(self, config: SandboxConfig | None = None) -> None:
        super().__init__(config or SandboxConfig(backend="docker"))
        self._worktree = GitWorktreeSandbox(self.config)

    def prepare(self, repo: Path, base_sha: str, run_id: str) -> Path:
        self.workspace = self._worktree.prepare(repo, base_sha, run_id)
        return self.workspace

    def sync_files(self, src: Path, rel_dest: str, *, read_only: bool = False) -> Path:
        return self._worktree.sync_files(src, rel_dest, read_only=read_only)

    def build_docker_command(
        self,
        cmd: list[str] | str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        readonly_mounts: list[tuple[Path, str]] | None = None,
    ) -> list[str]:
        """Build hardened ``docker run`` invocation with network/capability/resource bounds."""
        host_ws = (cwd or self.workspace or Path.cwd()).resolve()
        args: list[str] = [self.config.docker_bin, "run", "--rm"]
        if self.config.network_disabled:
            args.extend(["--network", "none"])
        if self.config.cpus > 0:
            args.extend(["--cpus", str(self.config.cpus)])
        if self.config.memory_mb > 0:
            args.extend(["--memory", f"{self.config.memory_mb}m"])
        if self.config.pids_limit > 0:
            args.extend(["--pids-limit", str(self.config.pids_limit)])
        args.extend(["--cap-drop=ALL", "--security-opt=no-new-privileges"])
        if self.config.read_only_root:
            args.extend(["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid"])
        args.extend(["-v", f"{host_ws}:/workspace:rw", "-w", "/workspace"])
        for host_path, container_path in readonly_mounts or []:
            args.extend(["-v", f"{Path(host_path).resolve()}:{container_path}:ro"])
        if env:
            for k, v in sorted(env.items()):
                args.extend(["-e", f"{k}={v}"])
        if self.config.extra_flags:
            args.extend(self.config.extra_flags)
        args.append(self.config.docker_image)
        if isinstance(cmd, str):
            args.extend(["sh", "-lc", cmd])
        else:
            args.extend([str(c) for c in cmd])
        return args

    def exec(
        self,
        cmd: list[str] | str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        readonly_mounts: list[tuple[Path, str]] | None = None,
    ) -> SandboxExecResult:
        docker_cmd = self.build_docker_command(
            cmd, cwd=cwd, env=env, readonly_mounts=readonly_mounts
        )
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return SandboxExecResult(
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                command=docker_cmd,
            )
        except subprocess.TimeoutExpired as exc:
            out_s = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err_s = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return SandboxExecResult(
                returncode=124,
                stdout=out_s,
                stderr=err_s,
                command=docker_cmd,
                timed_out=True,
            )

    def cleanup(self) -> None:
        self._worktree.cleanup()
        self.workspace = None


class RemoteSandboxAdapter(SandboxAdapter):
    """Extension hook for MicroVM or remote cloud sandbox backends."""

    backend = "remote"

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        driver: Any = None,
    ) -> None:
        super().__init__(config or SandboxConfig(backend="remote"))
        self.driver = driver if driver is not None else self._load_driver(self.config.remote_driver)

    @staticmethod
    def _load_driver(spec: str | None) -> Any:
        if not spec:
            return None
        if ":" in spec:
            mod_name, attr_name = spec.split(":", 1)
        elif "." in spec:
            mod_name, attr_name = spec.rsplit(".", 1)
        else:
            raise ValueError(f"Invalid remote_driver spec: {spec!r}")
        mod = importlib.import_module(mod_name)
        target = getattr(mod, attr_name)
        return target() if callable(target) else target

    def _require_driver(self) -> Any:
        if self.driver is None:
            raise NotImplementedError(
                "RemoteSandboxAdapter requires an explicit 'driver' or 'remote_driver' module:symbol."
            )
        return self.driver

    def prepare(self, repo: Path, base_sha: str, run_id: str) -> Path:
        drv = self._require_driver()
        self.workspace = Path(drv.prepare(repo, base_sha, run_id))
        return self.workspace

    def sync_files(self, src: Path, rel_dest: str, *, read_only: bool = False) -> Path:
        drv = self._require_driver()
        return Path(drv.sync_files(src, rel_dest, read_only=read_only))

    def exec(
        self,
        cmd: list[str] | str,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        readonly_mounts: list[tuple[Path, str]] | None = None,
    ) -> SandboxExecResult:
        drv = self._require_driver()
        res = drv.exec(cmd, cwd=cwd, env=env, timeout=timeout, readonly_mounts=readonly_mounts)
        if isinstance(res, SandboxExecResult):
            return res
        return SandboxExecResult(
            returncode=int(res.get("returncode", 0)),
            stdout=str(res.get("stdout", "")),
            stderr=str(res.get("stderr", "")),
            command=list(res.get("command", [])),
            timed_out=bool(res.get("timed_out", False)),
        )

    def cleanup(self) -> None:
        if self.driver is not None and hasattr(self.driver, "cleanup"):
            self.driver.cleanup()
        self.workspace = None


def get_sandbox_adapter(config: dict[str, Any] | None = None) -> SandboxAdapter:
    """Instantiate configured ``SandboxAdapter`` from full RSI config or sandbox sub-dict."""
    raw_cfg = config or {}
    sb_raw = (
        raw_cfg["sandbox"]
        if isinstance(raw_cfg.get("sandbox"), dict)
        else raw_cfg
    )
    sb_cfg = SandboxConfig.from_dict(sb_raw if isinstance(sb_raw, dict) else {})
    if sb_cfg.backend == "worktree":
        return GitWorktreeSandbox(sb_cfg)
    if sb_cfg.backend == "docker":
        return DockerSandbox(sb_cfg)
    if sb_cfg.backend in {"remote", "microvm"}:
        return RemoteSandboxAdapter(sb_cfg)
    raise ValueError(
        f"Unknown sandbox backend {sb_cfg.backend!r}. Expected 'worktree', 'docker', or 'remote'."
    )
