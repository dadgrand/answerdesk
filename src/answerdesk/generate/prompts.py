"""The prompt registry: versioned, hashed, renderable templates.

Requirement FR-GEN-04 asks for three things per prompt — an id, a semantic version and a
content hash — and §7.4 makes the prompt part of the answer's provenance. This module
enforces all three:

* ``configs/prompts/registry.yaml`` holds the templates as block scalars, each with
  ``id``, ``version`` and ``digest``;
* :func:`content_digest` hashes the normalised ``system`` + ``user`` body;
* :meth:`PromptRegistry.load` refuses to load a registry whose stored digest does not
  match the body, so "edit the prompt and forget the hash" is a hard error;
* :func:`stamp_registry` is the only writer: it recomputes digests and **refuses to
  keep a version number when the body changed**, which is the executable form of
  "changing a prompt without bumping the version fails the registry test".

Templating is a deliberately small ``{{ variable }}`` substitution rather than Jinja2:
the base install stays dependency-light (ADR-009) and the whole template language is
auditable in one function.

    # AT-GEN-04  # AT-GEN-11
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from answerdesk.contracts import Contract
from answerdesk.generate.errors import (
    PromptRegistryError,
    PromptRenderError,
    PromptVersionError,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default location of the registry (a code constant: no new gated config key).
REGISTRY_RELATIVE_PATH = Path("configs") / "prompts" / "registry.yaml"

#: Frozen stamp of (version, digest) per prompt — the version-bump guard.
LOCK_RELATIVE_PATH = Path("configs") / "prompts" / "registry.lock.yaml"

_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_DIGEST = re.compile(r"(?m)^(\s*digest:\s*)(\"[^\"]*\"|'[^']*'|\S+)\s*$")
_PROMPT_HEADER = re.compile(r"^(\s*)-\s+id:\s*(\S+)\s*$")


def repo_root() -> Path:
    """Repository root derived from this file (``src/answerdesk/generate/prompts.py``)."""
    return _REPO_ROOT


def default_registry_path() -> Path:
    return _REPO_ROOT / REGISTRY_RELATIVE_PATH


def default_lock_path() -> Path:
    return _REPO_ROOT / LOCK_RELATIVE_PATH


def normalise_text(text: str) -> str:
    """Canonical body text: no trailing spaces, no leading/trailing blank lines or padding."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def content_digest(system: str, user: str) -> str:
    """``sha256:<hex>`` over the normalised system and user bodies."""
    material = normalise_text(system) + "\x00" + normalise_text(user)
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def template_variables(template: str) -> set[str]:
    return set(_VARIABLE.findall(template))


def render_template(template: str, values: Mapping[str, Any], *, prompt_id: str = "") -> str:
    """Substitute ``{{ name }}`` with ``str(values[name])``; a missing name is an error."""
    missing = sorted({name for name in _VARIABLE.findall(template) if name not in values})
    if missing:
        raise PromptRenderError(
            f"prompt {prompt_id!r} needs variables {missing} that were not supplied",
            detail=f"supplied: {sorted(values)}",
        )
    return _VARIABLE.sub(lambda match: str(values[match.group(1)]), template)


class Prompt(Contract):
    """One versioned prompt. ``prompt_version`` is what every answer records."""

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    description: str = ""

    @property
    def prompt_version(self) -> str:
        """Stable provenance string, e.g. ``grounded-answer-1.0.0``."""
        return f"{self.id}-{self.version}"

    @property
    def variables(self) -> set[str]:
        return template_variables(self.system) | template_variables(self.user)

    def compute_digest(self) -> str:
        return content_digest(self.system, self.user)

    def render_system(self, **values: Any) -> str:
        return render_template(self.system, values, prompt_id=self.id)

    def render_user(self, **values: Any) -> str:
        return render_template(self.user, values, prompt_id=self.id)

    def check(self) -> None:
        """Raise if the stored digest no longer matches the body (FR-GEN-04)."""
        expected = self.compute_digest()
        if self.digest != expected:
            raise PromptRegistryError(
                f"prompt {self.id!r} v{self.version} digest mismatch: registry says "
                f"{self.digest!r}, body hashes to {expected!r}. Run "
                "`python -m answerdesk.generate --stamp` after bumping the version."
            )


class PromptLockEntry(Contract):
    """Frozen ``(version, digest)`` pair for one prompt."""

    version: str = Field(min_length=1)
    digest: str = Field(min_length=1)


class PromptLock(Contract):
    """The stamp file: what the registry looked like when it was last accepted."""

    registry_version: str = "1.0.0"
    prompts: dict[str, PromptLockEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> PromptLock | None:
        file_path = Path(path) if path is not None else default_lock_path()
        if not file_path.is_file():
            return None
        try:
            raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - filesystem dependent
            raise PromptRegistryError(f"cannot read prompt lock {file_path}: {exc}") from exc
        return cls.model_validate(raw or {})


class PromptRegistry:
    """An immutable set of prompts, loaded from YAML and verified on load."""

    def __init__(
        self,
        prompts: Mapping[str, Prompt],
        *,
        registry_version: str = "1.0.0",
        source_path: Path | None = None,
    ) -> None:
        if not prompts:
            raise PromptRegistryError("prompt registry is empty")
        self._prompts = dict(prompts)
        self.registry_version = registry_version
        self.source_path = source_path

    # -- construction -------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        verify: bool = True,
        lock_path: str | Path | None = None,
        verify_lock: bool = True,
    ) -> PromptRegistry:
        file_path = Path(path) if path is not None else default_registry_path()
        try:
            raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PromptRegistryError(f"cannot read prompt registry {file_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise PromptRegistryError(
                f"prompt registry {file_path} is not valid YAML: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise PromptRegistryError(f"prompt registry {file_path} must be a mapping")
        rows = raw.get("prompts")
        if not isinstance(rows, list) or not rows:
            raise PromptRegistryError(
                f"prompt registry {file_path} needs a non-empty 'prompts' list"
            )
        prompts: dict[str, Prompt] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise PromptRegistryError(
                    f"prompt registry {file_path}: each entry must be a mapping"
                )
            prompt = Prompt.model_validate(row)
            if prompt.id in prompts:
                raise PromptRegistryError(f"prompt registry {file_path} repeats id {prompt.id!r}")
            if verify:
                prompt.check()
            prompts[prompt.id] = prompt
        registry_version = str(raw.get("registry_version", "1.0.0"))
        if verify_lock:
            lock_file = (
                Path(lock_path)
                if lock_path is not None
                else file_path.parent / (LOCK_RELATIVE_PATH.name)
            )
            lock = PromptLock.load(lock_file)
            if lock is not None:
                _check_lock(prompts, lock, registry_version, lock_file)
        return cls(prompts, registry_version=registry_version, source_path=file_path)

    @classmethod
    def from_prompts(cls, prompts: Iterable[Prompt]) -> PromptRegistry:
        registry: dict[str, Prompt] = {}
        for prompt in prompts:
            prompt.check()
            registry[prompt.id] = prompt
        return cls(registry)

    # -- access -------------------------------------------------------------------
    def get(self, prompt_id: str) -> Prompt:
        prompt = self._prompts.get(prompt_id)
        if prompt is None:
            known = ", ".join(sorted(self._prompts))
            raise PromptRegistryError(f"unknown prompt id {prompt_id!r}; known ids: {known}")
        return prompt

    def prompt_version(self, prompt_id: str) -> str:
        return self.get(prompt_id).prompt_version

    @property
    def ids(self) -> list[str]:
        return sorted(self._prompts)

    def __contains__(self, prompt_id: object) -> bool:
        return prompt_id in self._prompts

    def __len__(self) -> int:
        return len(self._prompts)

    def digests(self) -> dict[str, str]:
        return {prompt_id: prompt.digest for prompt_id, prompt in sorted(self._prompts.items())}


# --------------------------------------------------------------------------------------
# Stamping: the only supported way to update the checked-in registry
# --------------------------------------------------------------------------------------


def _check_lock(
    prompts: Mapping[str, Prompt],
    lock: PromptLock,
    registry_version: str,
    lock_path: Path,
) -> None:
    """Refuse a registry whose content drifted from its stamped (version, digest)."""
    problems: list[str] = []
    if lock.registry_version != registry_version:
        problems.append(
            f"registry_version {registry_version!r} != stamped {lock.registry_version!r}"
        )
    for prompt_id, prompt in sorted(prompts.items()):
        entry = lock.prompts.get(prompt_id)
        if entry is None:
            problems.append(f"{prompt_id}: not present in the lock file")
            continue
        if entry.version != prompt.version:
            problems.append(f"{prompt_id}: version {prompt.version!r} != stamped {entry.version!r}")
        if entry.digest != prompt.digest:
            problems.append(f"{prompt_id}: digest differs from the stamped content hash")
    for prompt_id in sorted(set(lock.prompts) - set(prompts)):
        problems.append(f"{prompt_id}: stamped in the lock file but missing from the registry")
    if problems:
        raise PromptRegistryError(
            f"prompt registry does not match {lock_path.name} (FR-GEN-04): "
            + "; ".join(problems)
            + ". Bump the prompt version and run "
            "`python -m answerdesk.generate --stamp`.",
            detail="; ".join(problems),
        )


def _rewrite_digest(text: str, prompt_id: str, digest: str) -> str:
    """Replace the ``digest:`` line of one prompt block, leaving everything else intact."""
    lines = text.split("\n")
    in_block = False
    for index, line in enumerate(lines):
        header = _PROMPT_HEADER.match(line)
        if header is not None:
            in_block = header.group(2).strip().strip("\"'") == prompt_id
            continue
        if in_block and _DIGEST.match(line):
            indent = _DIGEST.match(line).group(1)  # type: ignore[union-attr]
            lines[index] = f'{indent}"{digest}"'
            return "\n".join(lines)
    raise PromptRegistryError(
        f"cannot stamp prompt {prompt_id!r}: no 'digest:' line found in its block"
    )


def _lock_text(prompts: Mapping[str, Prompt], registry_version: str) -> str:
    lines = [
        "# Prompt registry lock — generated by `python -m answerdesk.generate --stamp`.",
        "# Do not hand-edit: it records which (version, content hash) pair was reviewed.",
        f'registry_version: "{registry_version}"',
        "prompts:",
    ]
    for prompt_id, prompt in sorted(prompts.items()):
        lines.extend(
            [
                f"  {prompt_id}:",
                f'    version: "{prompt.version}"',
                f'    digest: "{prompt.digest}"',
            ]
        )
    return "\n".join(lines) + "\n"


def stamp_registry(
    path: str | Path | None = None,
    *,
    lock_path: str | Path | None = None,
    allow_stale_version: bool = False,
) -> dict[str, dict[str, str]]:
    """Recompute every digest, rewrite the ``digest:`` lines and refresh the lock file.

    Refuses (``PromptVersionError``) when a prompt body changed while its version did
    not — the "forgot to bump the version" defect FR-GEN-04 is about. The lock file is
    what makes that judgement possible: it remembers the version *and* hash that were
    last accepted. Returns ``{id: {"old", "new", "version", "changed"}}``.
    """
    file_path = Path(path) if path is not None else default_registry_path()
    lock_file = (
        Path(lock_path) if lock_path is not None else file_path.parent / (LOCK_RELATIVE_PATH.name)
    )
    text = file_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("prompts"), list):
        raise PromptRegistryError(f"prompt registry {file_path} has no 'prompts' list")
    registry_version = str(raw.get("registry_version", "1.0.0"))
    lock = PromptLock.load(lock_file)
    prompts: dict[str, Prompt] = {}
    changes: dict[str, dict[str, str]] = {}
    for row in raw["prompts"]:
        prompt = Prompt.model_validate(row)
        prompts[prompt.id] = prompt
        expected = prompt.compute_digest()
        previous = lock.prompts.get(prompt.id) if lock is not None else None
        if (
            previous is not None
            and previous.digest != expected
            and previous.version == prompt.version
            and not allow_stale_version
        ):
            raise PromptVersionError(
                f"prompt {prompt.id!r} body changed but the version is still "
                f"{prompt.version!r}: bump it (e.g. 1.0.0 -> 1.1.0) before stamping.",
                detail=f"stamped digest {previous.digest}, new digest {expected}",
            )
        if expected != prompt.digest:
            text = _rewrite_digest(text, prompt.id, expected)
        changes[prompt.id] = {
            "old": prompt.digest,
            "new": expected,
            "version": prompt.version,
            "changed": "true" if expected != prompt.digest else "false",
        }
        prompts[prompt.id] = prompt.model_copy(update={"digest": expected})
    if any(change["changed"] == "true" for change in changes.values()):
        file_path.write_text(text, encoding="utf-8")
    lock_file.write_text(_lock_text(prompts, registry_version), encoding="utf-8")
    return changes


def main(argv: list[str] | None = None) -> int:
    """``python -m answerdesk.generate --stamp`` — re-hash the registry."""
    import argparse

    parser = argparse.ArgumentParser(description="answerdesk prompt-registry tooling")
    parser.add_argument("--stamp", action="store_true", help="recompute digests and the lock file")
    parser.add_argument("--path", default=None, help="registry file (default configs/prompts)")
    parser.add_argument(
        "--allow-stale-version",
        action="store_true",
        help="rewrite digests even when a body changed without a version bump (escape hatch)",
    )
    args = parser.parse_args(argv)
    if not args.stamp:
        parser.print_help()
        return 2
    try:
        changes = stamp_registry(args.path, allow_stale_version=args.allow_stale_version)
    except PromptRegistryError as exc:
        print(f"{exc.code}: {exc.message}")
        return 1
    for prompt_id, change in sorted(changes.items()):
        state = "updated" if change["changed"] == "true" else "unchanged"
        print(f"{prompt_id} v{change['version']}: {state} ({change['new'][:19]}…)")
    return 0


if __name__ == "__main__":  # pragma: no cover - authoring entry point
    raise SystemExit(main())


__all__ = [
    "LOCK_RELATIVE_PATH",
    "REGISTRY_RELATIVE_PATH",
    "Prompt",
    "PromptLock",
    "PromptLockEntry",
    "PromptRegistry",
    "content_digest",
    "default_lock_path",
    "default_registry_path",
    "normalise_text",
    "render_template",
    "repo_root",
    "stamp_registry",
    "template_variables",
]
