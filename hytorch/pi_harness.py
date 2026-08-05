"""Private Pi runtime implementation backing ``hytorch.harness.pi``.

Pi (@earendil-works/pi-coding-agent) is the only harness for 0.1.0, configured
to call OpenAI models exclusively — no Anthropic model or key anywhere. Pi
is a Node.js SDK, not a CLI wrapped via shell-out: this runtime materializes a
small runtime script (hytorch/runtime/hytorch-pi.mjs, shipped inside this package)
plus its pinned npm dependency into a Docker image on first use, then runs
each node in a disposable container. An explicit host fallback is available
for development.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ._environment import agent_environment, docker_environment_file
from .harness import Result, Session, Usage

_DEFAULT_PROVIDER = "openai-codex"
_DEFAULT_MODEL = "gpt-5.6-terra"
_auth_lock = threading.Lock()


class PiRuntime:
    """Runs a node via Pi (@earendil-works/pi-coding-agent), the only
    harness in hytorch. Pi is configured to call OpenAI models exclusively:
    provider defaults to "openai-codex", Pi's OAuth provider for the
    operator's ChatGPT Plus/Pro (Codex) subscription (authenticated once
    via `pi` + `/login` -> "ChatGPT Plus/Pro (Codex)", credentials cached
    in ~/.pi/agent/auth.json and auto-refreshed — no API key needed).
    Setting provider to "openai" instead uses a plain OPENAI_API_KEY. No
    Anthropic model or key is ever selected. See AGENTS.md's Harness section.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str = _DEFAULT_MODEL,
        node_binary: str = "node",
        cache_dir: str | None = None,
        *,
        harness_name: str = "pi",
        docker: bool = True,
        docker_binary: str = "docker",
    ):
        self.provider = provider or _DEFAULT_PROVIDER
        self._provider_explicit = bool(provider)
        self.model = model or _DEFAULT_MODEL
        self.harness_name = harness_name
        self.node_binary = node_binary or "node"
        self.cache_dir = cache_dir
        self.docker = docker
        self.docker_binary = docker_binary or "docker"
        self._runtime_dir: str | None = None
        self._runtime_lock = threading.Lock()
        self._docker_image: str | None = None
        self._remote_docker: bool | None = None
        self._usage = Usage()
        self._usage_lock = threading.Lock()

    def usage(self) -> Usage:
        with self._usage_lock:
            return self._usage

    def start(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        session_dir = tempfile.mkdtemp(prefix="hytorch-pi-session-")
        try:
            text, session_id, session_file, _ = self._invoke(
                directory,
                prompt,
                mtype,
                session_dir=session_dir,
                temperature=temperature,
                max_tokens=max_tokens,
                read_only=read_only,
            )
        except Exception:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        return Result(
            text=text,
            session=Session(
                harness=self.harness_name, id=session_id, storage=session_file
            ),
        )

    def resume(
        self,
        session: Session,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> str:
        if session.harness != self.harness_name:
            raise ValueError(
                f"hytorch Pi harness cannot resume a {session.harness!r} session"
            )
        if not os.path.isfile(session.storage):
            raise RuntimeError(
                f"hytorch Pi harness: saved session {session.id!r} is unavailable"
            )
        text, resumed_id, _, _ = self._invoke(
            directory,
            prompt,
            mtype,
            session_dir=os.path.dirname(session.storage),
            session_file=session.storage,
            temperature=temperature,
            max_tokens=max_tokens,
            read_only=read_only,
        )
        if resumed_id != session.id:
            raise RuntimeError(
                f"hytorch Pi harness: resumed session {resumed_id!r}, expected {session.id!r}"
            )
        return text

    def close(self, session: Session) -> None:
        if session.harness != self.harness_name:
            raise ValueError(
                f"hytorch Pi harness cannot close a {session.harness!r} session"
            )
        shutil.rmtree(os.path.dirname(session.storage), ignore_errors=True)

    def _invoke(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        session_dir: str,
        session_file: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> tuple[str, str, str, Usage]:
        runtime_dir = self._ensure_runtime()
        resolved_model = mtype or self.model
        _validate_sampling(temperature, max_tokens)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="hytorch-pi-prompt-",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(prompt)
            prompt_path = f.name
        try:
            environment = agent_environment()
            provider = self._provider_for(environment)
            with docker_environment_file(values=environment) as environment_path:
                if self.docker:
                    runtime_name = self._docker_image or "hytorch-pi Docker image"
                    if self._uses_remote_docker():
                        result = self._run_remote_docker(
                            directory,
                            prompt_path,
                            resolved_model,
                            session_dir=session_dir,
                            session_file=session_file,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            read_only=read_only,
                            environment_path=environment_path,
                            provider=provider,
                        )
                    else:
                        with _staged_agent_directory() as agent_dir:
                            args = self._docker_run_args(
                                directory,
                                prompt_path,
                                resolved_model,
                                session_dir=session_dir,
                                session_file=session_file,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                read_only=read_only,
                                environment_path=environment_path,
                                provider=provider,
                                agent_dir=agent_dir,
                            )
                            result = subprocess.run(
                                args,
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                else:
                    script_path = os.path.join(runtime_dir, "hytorch-pi.mjs")
                    args = self._host_run_args(
                        script_path,
                        directory,
                        prompt_path,
                        resolved_model,
                        session_dir=session_dir,
                        session_file=session_file,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        provider=provider,
                    )
                    runtime_name = script_path
                    command_environment = dict(os.environ)
                    command_environment.update(environment)
                    result = subprocess.run(
                        args,
                        capture_output=True,
                        text=True,
                        check=False,
                        env=command_environment,
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"hytorch Pi harness: {runtime_name}: exit "
                        f"{result.returncode}: {result.stderr.strip()}"
                    )
                output = result.stdout.strip()
                if not output:
                    raise RuntimeError(
                        "hytorch Pi harness: runtime produced no output "
                        f"(stderr: {result.stderr.strip()})"
                    )
                try:
                    payload = json.loads(output)
                    text = payload["text"].strip()
                    session_id = payload["session_id"]
                    relative_file = payload["session_file"]
                    raw_usage = payload.get("usage", {})
                    usage = Usage(
                        input_tokens=int(raw_usage.get("input", 0)),
                        output_tokens=int(raw_usage.get("output", 0)),
                        cache_read_tokens=int(raw_usage.get("cacheRead", 0)),
                        cache_write_tokens=int(raw_usage.get("cacheWrite", 0)),
                    )
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"hytorch Pi harness: invalid runtime result: {output}"
                    ) from exc
                if not text or not session_id or not relative_file:
                    raise RuntimeError("hytorch Pi harness: incomplete runtime result")
                resolved_file = os.path.realpath(
                    os.path.join(session_dir, relative_file)
                )
                if os.path.commonpath(
                    (os.path.realpath(session_dir), resolved_file)
                ) != os.path.realpath(session_dir):
                    raise RuntimeError(
                        "hytorch Pi harness: session file escaped its storage directory"
                    )
                with self._usage_lock:
                    self._usage = Usage(
                        input_tokens=self._usage.input_tokens + usage.input_tokens,
                        output_tokens=self._usage.output_tokens + usage.output_tokens,
                        cache_read_tokens=(
                            self._usage.cache_read_tokens + usage.cache_read_tokens
                        ),
                        cache_write_tokens=(
                            self._usage.cache_write_tokens + usage.cache_write_tokens
                        ),
                    )
                return text, session_id, resolved_file, usage
        finally:
            os.remove(prompt_path)

    def _provider_for(self, environment: dict[str, str]) -> str:
        if not self._provider_explicit and "OPENAI_API_KEY" in environment:
            return "openai"
        return self.provider

    def _host_run_args(
        self,
        script_path: str,
        directory: str,
        prompt_path: str,
        model: str,
        *,
        session_dir: str,
        session_file: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
        provider: str | None = None,
    ) -> list[str]:
        args = [
            self.node_binary,
            script_path,
            "--cwd",
            directory,
            "--prompt-file",
            prompt_path,
            "--provider",
            provider or self.provider,
            "--model",
            model,
            "--session-dir",
            session_dir,
        ]
        if session_file is not None:
            args.extend(["--session-file", session_file])
        return _sampling_args(args, temperature, max_tokens)

    def _docker_run_args(
        self,
        directory: str,
        prompt_path: str,
        model: str,
        *,
        session_dir: str,
        session_file: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
        environment_path: str | None = None,
        provider: str | None = None,
        agent_dir: str | None = None,
    ) -> list[str]:
        if self._docker_image is None:
            raise RuntimeError("hytorch Pi harness: Docker image was not initialized")

        agent_dir = os.path.realpath(agent_dir or os.path.expanduser("~/.pi/agent"))
        os.makedirs(agent_dir, exist_ok=True)
        args = [
            self.docker_binary,
            "run",
            "--rm",
            "--init",
            "--mount",
            f"type=bind,src={os.path.realpath(directory)},dst=/workspace",
            "--mount",
            f"type=bind,src={os.path.realpath(prompt_path)},dst=/run/hytorch/prompt.txt,readonly",
            "--mount",
            f"type=bind,src={agent_dir},dst=/root/.pi/agent",
            "--mount",
            f"type=bind,src={os.path.realpath(session_dir)},dst=/run/hytorch/session",
        ]
        if environment_path is not None:
            args.extend(["--env-file", os.path.realpath(environment_path)])
        root = os.path.realpath(directory)
        for path in read_only:
            resolved = os.path.realpath(path)
            if os.path.commonpath((root, resolved)) != root:
                raise ValueError(
                    "hytorch Pi harness: read-only path escaped the node root"
                )
            relative = os.path.relpath(resolved, root)
            target = os.path.join("/workspace", relative)
            args.extend(["--mount", f"type=bind,src={resolved},dst={target},readonly"])
        args.extend(
            [
                self._docker_image,
                "--cwd",
                "/workspace",
                "--prompt-file",
                "/run/hytorch/prompt.txt",
                "--provider",
                provider or self.provider,
                "--model",
                model,
                "--session-dir",
                "/run/hytorch/session",
            ]
        )
        if session_file is not None:
            args.extend(
                [
                    "--session-file",
                    f"/run/hytorch/session/{os.path.basename(session_file)}",
                ]
            )
        return _sampling_args(args, temperature, max_tokens)

    def _uses_remote_docker(self) -> bool:
        if self._remote_docker is not None:
            return self._remote_docker
        host = os.environ.get("DOCKER_HOST")
        if host is None:
            command = [
                self.docker_binary,
                "context",
                "inspect",
                "--format",
                '{{(index .Endpoints "docker").Host}}',
            ]
            context = os.environ.get("DOCKER_CONTEXT")
            if context:
                command.append(context)
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            host = result.stdout.strip() if result.returncode == 0 else ""
        self._remote_docker = bool(host) and not host.startswith(
            ("unix://", "npipe://", "/")
        )
        return self._remote_docker

    def _run_remote_docker(
        self,
        directory: str,
        prompt_path: str,
        model: str,
        *,
        session_dir: str,
        session_file: str | None,
        temperature: float | None,
        max_tokens: int | None,
        read_only: tuple[str, ...],
        environment_path: str | None,
        provider: str,
    ) -> subprocess.CompletedProcess[str]:
        if self._docker_image is None:
            raise RuntimeError("hytorch Pi harness: Docker image was not initialized")
        if provider == "openai-codex":
            raise RuntimeError(
                "hytorch Pi harness: remote Docker contexts require OPENAI_API_KEY; "
                "local Pi login bind mounts are not available remotely"
            )
        prefix = f"hytorch-{uuid.uuid4().hex}"
        volumes = {
            "statespace": prefix + "-state",
            "workspace": prefix + "-workspace",
            "session": prefix + "-session",
            "prompt": prefix + "-prompt",
        }
        created: list[str] = []
        prompt_dir = tempfile.mkdtemp(prefix="hytorch-prompt-stage-")
        staged_prompt = os.path.join(prompt_dir, "prompt.txt")
        shutil.copy2(prompt_path, staged_prompt)
        try:
            for volume in volumes.values():
                result = subprocess.run(
                    [self.docker_binary, "volume", "create", volume],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"hytorch Pi harness: create remote volume: {result.stderr.strip()}"
                    )
                created.append(volume)
            self._upload_volume(
                os.path.join(directory, "statespace"), volumes["statespace"]
            )
            self._upload_volume(
                os.path.join(directory, "workspace"), volumes["workspace"]
            )
            self._upload_volume(session_dir, volumes["session"])
            self._upload_volume(prompt_dir, volumes["prompt"])

            root = os.path.realpath(directory)
            read_only_paths = {os.path.realpath(path) for path in read_only}
            if any(
                os.path.commonpath((root, path)) != root for path in read_only_paths
            ):
                raise ValueError(
                    "hytorch Pi harness: read-only path escaped the node root"
                )
            args = [self.docker_binary, "run", "--rm", "--init"]
            for name in ("statespace", "workspace"):
                source = os.path.realpath(os.path.join(directory, name))
                option = f"type=volume,src={volumes[name]},dst=/workspace/{name}" + (
                    ",readonly" if source in read_only_paths else ""
                )
                args.extend(["--mount", option])
            args.extend(
                [
                    "--mount",
                    f"type=volume,src={volumes['session']},dst=/run/hytorch/session",
                    "--mount",
                    f"type=volume,src={volumes['prompt']},dst=/run/hytorch/prompt,readonly",
                ]
            )
            if environment_path is not None:
                args.extend(["--env-file", os.path.realpath(environment_path)])
            args.extend(
                [
                    self._docker_image,
                    "--cwd",
                    "/workspace",
                    "--prompt-file",
                    "/run/hytorch/prompt/prompt.txt",
                    "--provider",
                    provider,
                    "--model",
                    model,
                    "--session-dir",
                    "/run/hytorch/session",
                ]
            )
            if session_file is not None:
                args.extend(
                    [
                        "--session-file",
                        f"/run/hytorch/session/{os.path.basename(session_file)}",
                    ]
                )
            result = subprocess.run(
                _sampling_args(args, temperature, max_tokens),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for name in ("statespace", "workspace"):
                    source = os.path.realpath(os.path.join(directory, name))
                    if source not in read_only_paths:
                        self._download_volume(volumes[name], source)
                self._download_volume(volumes["session"], session_dir)
            return result
        finally:
            shutil.rmtree(prompt_dir, ignore_errors=True)
            for volume in reversed(created):
                subprocess.run(
                    [self.docker_binary, "volume", "rm", "--force", volume],
                    capture_output=True,
                    text=True,
                    check=False,
                )

    def _upload_volume(self, source: str, volume: str) -> None:
        archive_path = _archive_directory(source)
        try:
            with open(archive_path, "rb") as archive:
                result = subprocess.run(
                    [
                        self.docker_binary,
                        "run",
                        "--rm",
                        "--interactive",
                        "--mount",
                        f"type=volume,src={volume},dst=/data",
                        "--entrypoint",
                        "tar",
                        self._docker_image,
                        "-C",
                        "/data",
                        "-xf",
                        "-",
                    ],
                    stdin=archive,
                    capture_output=True,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    "hytorch Pi harness: upload remote node state: "
                    + result.stderr.decode(errors="replace").strip()
                )
        finally:
            os.remove(archive_path)

    def _download_volume(self, volume: str, destination: str) -> None:
        descriptor, archive_path = tempfile.mkstemp(
            prefix="hytorch-download-", suffix=".tar"
        )
        try:
            with os.fdopen(descriptor, "wb") as archive:
                result = subprocess.run(
                    [
                        self.docker_binary,
                        "run",
                        "--rm",
                        "--mount",
                        f"type=volume,src={volume},dst=/data,readonly",
                        "--entrypoint",
                        "tar",
                        self._docker_image,
                        "-C",
                        "/data",
                        "-cf",
                        "-",
                        ".",
                    ],
                    stdout=archive,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    "hytorch Pi harness: download remote node state: "
                    + result.stderr.decode(errors="replace").strip()
                )
            _replace_from_archive(archive_path, destination)
        finally:
            try:
                os.remove(archive_path)
            except FileNotFoundError:
                pass

    def _ensure_runtime(self) -> str:
        if self._runtime_dir is not None:
            return self._runtime_dir
        with self._runtime_lock:
            if self._runtime_dir is None:
                self._runtime_dir = self._materialize_runtime()
        return self._runtime_dir

    def _materialize_runtime(self) -> str:
        # __package__ is typed str | None (it's only ever None for a
        # top-level script run directly, not a module imported as part of a
        # package), but this module is always imported as hytorch.pi_harness, so
        # it's always "hytorch" here. The "hytorch" fallback exists only to satisfy
        # the type checker (importlib.resources.files requires a
        # str-or-module, never None) without changing behavior.
        package = __package__ or "hytorch"
        runtime = importlib.resources.files(package) / "runtime"
        script_bytes = (runtime / "hytorch-pi.mjs").read_bytes()
        package_json_bytes = (runtime / "package.json").read_bytes()
        dockerfile_bytes = (runtime / "Dockerfile").read_bytes()
        digest = hashlib.sha256(
            script_bytes + package_json_bytes + dockerfile_bytes
        ).hexdigest()[:16]

        directory = self.cache_dir
        if directory is None:
            base = _user_cache_dir()
            # Namespace by all runtime inputs so code or dependency changes
            # get a fresh image/install rather than silently reusing stale bits.
            directory = os.path.join(base, "hytorch", "pi-runtime", digest)

        os.makedirs(directory, exist_ok=True)
        (Path(directory) / "hytorch-pi.mjs").write_bytes(script_bytes)
        (Path(directory) / "package.json").write_bytes(package_json_bytes)
        (Path(directory) / "Dockerfile").write_bytes(dockerfile_bytes)

        if self.docker:
            self._ensure_docker_image(directory, digest)
            return directory

        node_modules = (
            Path(directory) / "node_modules" / "@earendil-works" / "pi-coding-agent"
        )
        if node_modules.exists():
            return directory

        result = subprocess.run(
            ["npm", "install", "--no-audit", "--no-fund"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"hytorch Pi harness: npm install in {directory}: {result.stderr.strip()}"
            )
        return directory

    def _ensure_docker_image(self, directory: str, digest: str) -> None:
        configured_image = os.environ.get("HYTORCH_PI_IMAGE")
        image = configured_image or f"hytorch-pi-runtime:{digest}"
        inspect = subprocess.run(
            [self.docker_binary, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect.returncode != 0 and configured_image:
            raise RuntimeError(
                f"hytorch Pi harness: HYTORCH_PI_IMAGE {image!r} is not available "
                "in the active Docker context"
            )
        if inspect.returncode != 0:
            build = subprocess.run(
                [self.docker_binary, "build", "--tag", image, directory],
                capture_output=True,
                text=True,
                check=False,
            )
            if build.returncode != 0:
                raise RuntimeError(
                    f"hytorch Pi harness: build Docker image {image}: {build.stderr.strip()}"
                )
            inspect = subprocess.run(
                [
                    self.docker_binary,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    image,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        resolved = inspect.stdout.strip()
        if inspect.returncode != 0 or not resolved:
            raise RuntimeError(
                f"hytorch Pi harness: cannot resolve Docker image {image!r}"
            )
        self._docker_image = resolved


@contextmanager
def _staged_agent_directory() -> Iterator[str]:
    """Give one container an isolated Pi config and preserve newer OAuth data."""
    source = os.path.realpath(os.path.expanduser("~/.pi/agent"))
    os.makedirs(source, exist_ok=True)
    parent = tempfile.mkdtemp(prefix="hytorch-pi-agent-")
    staged = os.path.join(parent, "agent")
    with _auth_lock:
        shutil.copytree(source, staged)
    try:
        yield staged
    finally:
        with _auth_lock:
            _promote_newer_auth(os.path.join(staged, "auth.json"), source)
        shutil.rmtree(parent, ignore_errors=True)


def _promote_newer_auth(candidate: str, destination_dir: str) -> None:
    """Promote refreshed credentials without letting an older worker win."""
    if not os.path.isfile(candidate):
        return
    destination = os.path.join(destination_dir, "auth.json")
    try:
        candidate_data = json.loads(Path(candidate).read_text(encoding="utf-8"))
        current_data = (
            json.loads(Path(destination).read_text(encoding="utf-8"))
            if os.path.isfile(destination)
            else {}
        )
    except (OSError, json.JSONDecodeError):
        return
    changed = False
    for provider, value in candidate_data.items():
        if not isinstance(value, dict):
            continue
        current = current_data.get(provider)
        candidate_expiry = value.get("expires", 0)
        current_expiry = current.get("expires", 0) if isinstance(current, dict) else 0
        if provider not in current_data or candidate_expiry > current_expiry:
            current_data[provider] = value
            changed = True
    if not changed:
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix="auth-", suffix=".json", dir=destination_dir
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(current_data, file, indent=2)
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _user_cache_dir() -> str:
    if os.name == "nt":  # pragma: no cover - not a supported target
        return os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return xdg
    if os.uname().sysname == "Darwin":
        return os.path.expanduser("~/Library/Caches")
    return os.path.expanduser("~/.cache")


def _archive_directory(source: str) -> str:
    descriptor, path = tempfile.mkstemp(prefix="hytorch-upload-", suffix=".tar")
    os.close(descriptor)
    try:
        with tarfile.open(path, "w") as archive:
            for name in sorted(os.listdir(source)):
                archive.add(os.path.join(source, name), arcname=name, recursive=True)
        return path
    except Exception:
        os.remove(path)
        raise


def _replace_from_archive(archive_path: str, destination: str) -> None:
    parent = os.path.dirname(os.path.realpath(destination))
    extracted = tempfile.mkdtemp(prefix="hytorch-extract-", dir=parent)
    try:
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                normalized = os.path.normpath(member.name)
                if (
                    os.path.isabs(member.name)
                    or normalized == ".."
                    or normalized.startswith(".." + os.sep)
                ):
                    raise RuntimeError(
                        "hytorch Pi harness: remote archive contains an unsafe path"
                    )
                if member.issym() or member.islnk():
                    target = os.path.normpath(member.linkname)
                    if (
                        os.path.isabs(member.linkname)
                        or target == ".."
                        or target.startswith(".." + os.sep)
                    ):
                        raise RuntimeError(
                            "hytorch Pi harness: remote archive contains an unsafe link"
                        )
            archive.extractall(extracted, filter="data")
        if os.path.isdir(destination):
            for current, directories, _ in os.walk(destination):
                os.chmod(current, 0o700)
                for name in directories:
                    os.chmod(os.path.join(current, name), 0o700)
            shutil.rmtree(destination)
        elif os.path.exists(destination):
            os.remove(destination)
        os.replace(extracted, destination)
        extracted = ""
    finally:
        if extracted:
            shutil.rmtree(extracted, ignore_errors=True)


def _validate_sampling(temperature: float | None, max_tokens: int | None) -> None:
    if temperature is not None and (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature < 0
    ):
        raise ValueError(
            "hytorch Pi harness: temperature must be a non-negative number or None"
        )
    if max_tokens is not None and (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError(
            "hytorch Pi harness: max_tokens must be a positive integer or None"
        )


def _sampling_args(
    args: list[str], temperature: float | None, max_tokens: int | None
) -> list[str]:
    if temperature is not None:
        args.extend(["--temperature", str(temperature)])
    if max_tokens is not None:
        args.extend(["--max-tokens", str(max_tokens)])
    return args
