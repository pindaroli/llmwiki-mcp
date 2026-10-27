#!/usr/bin/env python3
"""
Estensione nativa k8s-lab per il fork di llmwiki (Opzione A - Esecuzione Locale stdio).
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml as _yaml_mod  # type: ignore[import-untyped]
except ModuleNotFoundError:
    _yaml_mod = None

ACTIVE_PLAN_STATUSES: frozenset[str] = frozenset({"active", "approved", "in_progress"})
FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)$", re.DOTALL)
CJK_SECTION_RE = re.compile(
    r"(?:^|\n)##\s+(?:中文|日本語)\b.*?(?=\n##\s+(?!中文|日本語)|\Z)",
    re.DOTALL,
)

DEFAULT_CONTENT_DIRS: dict[str, str] = {
    "concept": "wiki/concepts",
    "entity": "wiki/entities",
    "pattern": "wiki/patterns",
    "workflow": "wiki/workflows",
    "procedure": "wiki/procedures",
    "plan": "wiki/plans",
}


def _unquote_scalar(val: str) -> str:
    val = val.strip()
    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
        return val[1:-1]
    return val


def _stdlib_parse_yaml_frontmatter(raw_yaml: str) -> dict[str, Any]:
    """
    Parser YAML frontmatter leggero basato solo sulla libreria standard Python.
    Usato automaticamente come fallback quando PyYAML non è installato nel Python di sistema.
    """
    meta: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in raw_yaml.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and current_list_key is not None:
            item_val = _unquote_scalar(stripped[2:].strip())
            if isinstance(meta.get(current_list_key), list):
                meta[current_list_key].append(item_val)
            continue

        if ":" in raw_line and not raw_line.startswith((" ", "\t")):
            key_part, val_part = raw_line.split(":", 1)
            key = key_part.strip()
            val_str = val_part.strip()
            if not key:
                continue

            if not val_str:
                meta[key] = []
                current_list_key = key
                continue

            current_list_key = None
            if val_str.startswith("[") and val_str.endswith("]"):
                inner = val_str[1:-1].strip()
                if not inner:
                    meta[key] = []
                else:
                    meta[key] = [_unquote_scalar(x) for x in inner.split(",") if x.strip()]
            elif val_str.lower() in ("true", "false"):
                meta[key] = val_str.lower() == "true"
            else:
                meta[key] = _unquote_scalar(val_str)

    return meta


def get_project_root(cfg: dict[str, Any] | None = None) -> Path:
    """Risolve la root del progetto da cfg['_root'] o dalla directory corrente."""
    if cfg and "_root" in cfg and cfg["_root"]:
        return Path(cfg["_root"]).resolve()
    env_cfg = os.environ.get("LLMWIKI_CONFIG")
    if env_cfg:
        return Path(env_cfg).resolve().parent
    return Path.cwd().resolve()


def get_content_dirs(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    """
    Restituisce la mappa ordinata `{categoria: Path_assoluto}` per tutte le directory
    della Knowledge Base configurate in `paths.content_dirs`.
    """
    base_root = root or get_project_root(cfg)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    raw_content_dirs = paths_cfg.get("content_dirs")

    if isinstance(raw_content_dirs, dict) and raw_content_dirs:
        resolved: dict[str, Path] = {}
        for category, rel_path in raw_content_dirs.items():
            if rel_path:
                resolved[str(category).strip()] = (base_root / str(rel_path)).resolve()
        if resolved:
            return resolved

    concepts_rel = paths_cfg.get("concepts", "wiki/concepts")
    return {"concept": (base_root / str(concepts_rel)).resolve()}


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Estrae e parsifica il frontmatter YAML e il corpo Markdown (con fallback stdlib)."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml, body = match.group(1), match.group(2)
    if _yaml_mod is not None:
        try:
            loaded = _yaml_mod.safe_load(raw_yaml)
            meta = loaded if isinstance(loaded, dict) else {}
        except Exception:
            meta = _stdlib_parse_yaml_frontmatter(raw_yaml)
    else:
        meta = _stdlib_parse_yaml_frontmatter(raw_yaml)
    return meta, body


def normalize_k8slab_frontmatter(
    meta: dict[str, Any],
    default_category: str = "concept",
    md_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Normalizza in memoria i metadati secondo le convenzioni di k8s-lab senza
    alterare il file Markdown su disco.
    """
    norm = dict(meta) if isinstance(meta, dict) else {}

    if md_path is not None and not norm.get("title"):
        norm["title"] = md_path.stem.replace("-", " ").title()

    raw_cat = norm.get("category") or norm.get("type") or default_category
    norm["category"] = str(raw_cat).strip().lower()
    norm.setdefault("type", norm["category"])

    if "updated" not in norm:
        if "last_updated" in norm and norm["last_updated"] is not None:
            norm["updated"] = str(norm["last_updated"])
        elif "created_at" in norm and norm["created_at"] is not None:
            norm["updated"] = str(norm["created_at"])

    raw_tags = norm.get("tags")
    if isinstance(raw_tags, list):
        cleaned_tags: list[str] = []
        for tag in raw_tags:
            tag_str = str(tag).strip().lstrip("#").strip()
            if tag_str and tag_str not in cleaned_tags:
                cleaned_tags.append(tag_str)
        norm["tags"] = cleaned_tags
    elif isinstance(raw_tags, str):
        norm["tags"] = [
            t.strip().lstrip("#")
            for t in raw_tags.split(",")
            if t.strip().lstrip("#")
        ]
    else:
        norm.setdefault("tags", [])

    if md_path is not None and root is not None:
        with contextlib.suppress(ValueError):
            norm["source_path"] = str(md_path.resolve().relative_to(root.resolve()))

    return norm


def is_active_kb_document(
    md_path: Path,
    category: str,
    meta: dict[str, Any] | None = None,
) -> bool:
    """
    Verifica se un file Markdown deve essere indicizzato da llmwiki:
    - Ignora README.md e i file che iniziano con `_` o `.`
    - Per la categoria `plan` (`wiki/plans/`), include solo i piani con
      `status` in {"active", "approved", "in_progress"}.
    """
    name = md_path.name
    if name.startswith("_") or name.startswith(".") or name.lower() == "readme.md":
        return False

    if meta is None:
        try:
            meta, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
        except OSError:
            return False

    status = str(meta.get("status", "active")).strip().lower()
    if category == "plan" or md_path.parent.name == "plans":
        return status in ACTIVE_PLAN_STATUSES

    return status not in {"archived", "deprecated", "retired"}


def iter_kb_markdown_files(
    cfg: dict[str, Any],
    root: Path | None = None,
    include_outputs: bool = False,
) -> Iterator[tuple[str, Path, dict[str, Any], str]]:
    """
    Itera deterministicamente su tutti i file Markdown attivi della Knowledge Base
    attraverso tutte le `content_dirs` configurate.
    """
    base_root = root or get_project_root(cfg)
    content_dirs = get_content_dirs(cfg, base_root)
    seen_paths: set[Path] = set()

    for category, folder_path in content_dirs.items():
        if not folder_path.is_dir():
            continue
        for md_path in sorted(folder_path.glob("*.md")):
            resolved_path = md_path.resolve()
            if resolved_path in seen_paths:
                continue
            try:
                raw_text = resolved_path.read_text(encoding="utf-8")
            except OSError:
                continue

            raw_meta, body = parse_frontmatter(raw_text)
            if not is_active_kb_document(resolved_path, category, raw_meta):
                continue

            norm_meta = normalize_k8slab_frontmatter(
                raw_meta,
                default_category=category,
                md_path=resolved_path,
                root=base_root,
            )
            seen_paths.add(resolved_path)
            yield norm_meta["category"], resolved_path, norm_meta, body

    if include_outputs:
        outputs_rel = cfg.get("paths", {}).get("outputs", "wiki/outputs")
        outputs_dir = (base_root / str(outputs_rel)).resolve()
        if outputs_dir.is_dir():
            for md_path in sorted(outputs_dir.glob("*.md")):
                resolved_path = md_path.resolve()
                if resolved_path in seen_paths or md_path.name.startswith("_"):
                    continue
                try:
                    raw_text = resolved_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                raw_meta, body = parse_frontmatter(raw_text)
                norm_meta = normalize_k8slab_frontmatter(
                    raw_meta,
                    default_category="output",
                    md_path=resolved_path,
                    root=base_root,
                )
                seen_paths.add(resolved_path)
                yield "output", resolved_path, norm_meta, body


def list_kb_paths(
    cfg: dict[str, Any],
    root: Path | None = None,
    include_outputs: bool = False,
) -> list[Path]:
    """Restituisce la lista ordinata di tutti i `Path` attivi nella Knowledge Base."""
    return [
        md_path
        for _, md_path, _, _ in iter_kb_markdown_files(
            cfg, root=root, include_outputs=include_outputs
        )
    ]


def resolve_slug_path(
    cfg: dict[str, Any],
    slug: str,
    root: Path | None = None,
    category: str | None = None,
    for_write: bool = False,
) -> Path:
    """
    Risolve uno `slug` nel suo `Path` reale attraverso tutte le `content_dirs`.
    """
    clean_slug = slug.strip()
    if clean_slug.endswith(".md"):
        clean_slug = clean_slug[:-3]
    clean_slug = Path(clean_slug).name

    base_root = root or get_project_root(cfg)
    content_dirs = get_content_dirs(cfg, base_root)

    for cat, folder_path in content_dirs.items():
        candidate = folder_path / f"{clean_slug}.md"
        if candidate.is_file():
            if not for_write and not is_active_kb_document(candidate, cat):
                continue
            return candidate

    if category:
        cat_key = category.strip().lower().rstrip("s")
        for registered_cat, folder_path in content_dirs.items():
            if registered_cat.lower().rstrip("s") == cat_key:
                if for_write:
                    folder_path.mkdir(parents=True, exist_ok=True)
                return folder_path / f"{clean_slug}.md"

    default_dir = content_dirs.get("concept") or next(iter(content_dirs.values()))
    if for_write:
        default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"{clean_slug}.md"


def strip_cjk_sections(text: str) -> str:
    """Rimuove le sezioni trilingui `## 中文` e `## 日本語` lasciando intatto il resto."""
    return CJK_SECTION_RE.sub("", text).rstrip() + "\n"


@contextlib.contextmanager
def meta_write_lock(cfg: dict[str, Any], root: Path | None = None) -> Iterator[Path]:
    """
    Acquisisce un file lock esclusivo (`fcntl.flock`) su `wiki/_meta/.lock`.
    """
    base_root = root or get_project_root(cfg)
    meta_rel = cfg.get("paths", {}).get("meta", "wiki/_meta")
    meta_dir = (base_root / str(meta_rel)).resolve()
    meta_dir.mkdir(parents=True, exist_ok=True)
    lock_file = meta_dir / ".lock"

    with open(lock_file, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield meta_dir
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def atomic_write_text(target_path: Path, content: str, encoding: str = "utf-8") -> None:
    """Scrive `target_path` in modo atomico usando un file temporaneo nella stessa directory."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as tmp_fh:
            tmp_fh.write(content)
            tmp_fh.flush()
            os.fsync(tmp_fh.fileno())
        os.replace(tmp_name, target_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
