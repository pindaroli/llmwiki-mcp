"""
llmwiki.k8slab_dirs — Modulo core per il fork k8s-lab di llmwiki (v0.9.1-k8slab).

Funzionalità fornite:
  1. `get_content_dirs(cfg, root)`: risolve la mappa `{category: Path}` da `paths.content_dirs`.
  2. `iter_kb_markdown_files(cfg, root, include_outputs)`: scansiona tutte le cartelle
     semantiche di k8s-lab (`concepts`, `entities`, `patterns`, `workflows`, `procedures`,
     `plans`), filtrando per `plans` solo i documenti con `status: active | approved | in_progress`.
  3. `resolve_slug_path(cfg, slug, category, root, for_write)`: risolve qualsiasi `[[slug]]`
     attraverso tutte le cartelle configurate e indirizza i nuovi articoli nella cartella
     della categoria corrispondente.
  4. `normalize_k8slab_frontmatter(...)`: traduce al volo `type` -> `category`,
     `last_updated` -> `updated` e rimuove il prefisso `#` dai tag.
  5. `meta_write_lock(cfg, root)` + `atomic_write_text(path, content)`: file-locking esclusivo
     con `fcntl.flock` su `wiki/_meta/.lock` e scrittura atomica (`tempfile` + `os.replace`).
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

import yaml

ACTIVE_PLAN_STATUSES = {"active", "approved", "in_progress"}
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
CJK_SECTION_RE = re.compile(
    r"\n##\s+(?:中文|日本語)\b.*?(?=\n##\s+(?!中文|日本語)|\Z)",
    re.DOTALL,
)


def get_project_root(cfg: dict[str, Any] | None = None) -> Path:
    """Risolve la root del progetto a partire da cfg['_root'], LLMWIKI_CONFIG o CWD."""
    if cfg and isinstance(cfg.get("_root"), (str, Path)):
        return Path(cfg["_root"]).resolve()
    env_cfg = os.environ.get("LLMWIKI_CONFIG")
    if env_cfg:
        cfg_path = Path(env_cfg).resolve()
        if cfg_path.is_file():
            return cfg_path.parent
    return Path.cwd().resolve()


def get_content_dirs(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    """
    Restituisce la mappa ordinata `{category: Path}` delle directory Markdown abilitate.
    Se `paths.content_dirs` è presente in `config.yaml`, usa quelle directory;
    altrimenti mantiene il fallback compatibile con upstream (`{"concept": paths.concepts}`).
    """
    base = root or get_project_root(cfg)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    raw_content_dirs = paths_cfg.get("content_dirs")

    resolved: dict[str, Path] = {}
    if isinstance(raw_content_dirs, dict) and raw_content_dirs:
        for category, rel_path in raw_content_dirs.items():
            p = Path(rel_path)
            resolved[str(category)] = p if p.is_absolute() else (base / p)
        return resolved

    concepts_rel = paths_cfg.get("concepts", "wiki/concepts")
    cp = Path(concepts_rel)
    resolved["concept"] = cp if cp.is_absolute() else (base / cp)
    return resolved


def parse_markdown_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Estrae `(frontmatter_dict, body)` da un documento Markdown."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml, body = match.group(1), match.group(2)
    try:
        loaded = yaml.safe_load(raw_yaml)
        meta = loaded if isinstance(loaded, dict) else {}
    except Exception:
        meta = {}
    return meta, body


def normalize_k8slab_frontmatter(
    meta: dict[str, Any],
    default_category: str = "concept",
    slug: str | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """
    Normalizza in modo non distruttivo i metadati di k8s-lab verso lo schema atteso da llmwiki:
      - `type` -> `category` (se `category` è assente)
      - `last_updated` o `created_at` -> `updated` (se `updated` è assente)
      - rimuove il prefisso `#` da ogni elemento di `tags`
      - garantisce sempre la presenza di `title`
    """
    norm = dict(meta) if isinstance(meta, dict) else {}

    if not norm.get("category"):
        norm["category"] = str(norm.get("type") or default_category)

    if not norm.get("updated"):
        if norm.get("last_updated"):
            norm["updated"] = str(norm["last_updated"])
        elif norm.get("created_at"):
            norm["updated"] = str(norm["created_at"])

    if not norm.get("title") and slug:
        norm["title"] = slug.replace("-", " ").replace("_", " ").title()

    raw_tags = norm.get("tags")
    if isinstance(raw_tags, list):
        norm["tags"] = [str(t).lstrip("#").strip() for t in raw_tags if str(t).strip()]
    elif isinstance(raw_tags, str):
        norm["tags"] = [t.lstrip("#").strip() for t in raw_tags.split(",") if t.strip()]

    if source_path is not None and "source_path" not in norm:
        norm["source_path"] = str(source_path)

    return norm


def is_active_kb_document(
    md_path: Path,
    category: str,
    meta: dict[str, Any] | None = None,
) -> bool:
    """
    Determina se un file Markdown deve essere indicizzato:
      - Ignora `README.md` e i file che iniziano con `_` o `.`.
      - Per la categoria `plan` (o cartella `plans`), include SOLO i piani con
        `status` in `{"active", "approved", "in_progress"}`.
    """
    name = md_path.name
    if name.startswith("_") or name.startswith(".") or name.lower() == "readme.md":
        return False

    if category == "plan" or md_path.parent.name == "plans":
        if meta is None:
            try:
                meta, _ = parse_markdown_frontmatter(md_path.read_text(encoding="utf-8"))
            except Exception:
                return False
        status = str(meta.get("status", "")).strip().lower()
        return status in ACTIVE_PLAN_STATUSES

    return True


def iter_kb_markdown_files(
    cfg: dict[str, Any],
    root: Path | None = None,
    include_outputs: bool = False,
) -> list[Path]:
    """
    Scansiona deterministicamente tutte le cartelle configurate in `paths.content_dirs`
    (più `paths.outputs` se `include_outputs=True`), filtrando i piani non attivi.
    """
    base = root or get_project_root(cfg)
    content_dirs = get_content_dirs(cfg, base)
    collected: list[Path] = []
    seen_paths: set[Path] = set()

    for category, folder in content_dirs.items():
        if not folder.is_dir():
            continue
        for md_file in sorted(folder.glob("*.md")):
            resolved = md_file.resolve()
            if resolved in seen_paths:
                continue
            if not is_active_kb_document(md_file, category):
                continue
            seen_paths.add(resolved)
            collected.append(md_file)

    if include_outputs:
        outputs_rel = cfg.get("paths", {}).get("outputs", "wiki/outputs")
        outputs_dir = Path(outputs_rel)
        if not outputs_dir.is_absolute():
            outputs_dir = base / outputs_dir
        if outputs_dir.is_dir():
            for md_file in sorted(outputs_dir.glob("*.md")):
                resolved = md_file.resolve()
                if resolved not in seen_paths and not md_file.name.startswith("_"):
                    seen_paths.add(resolved)
                    collected.append(md_file)

    return collected


def detect_category_for_path(cfg: dict[str, Any], md_path: Path, root: Path | None = None) -> str:
    """Restituisce la categoria associata alla directory padre di `md_path`."""
    base = root or get_project_root(cfg)
    resolved_parent = md_path.resolve().parent
    for category, folder in get_content_dirs(cfg, base).items():
        if folder.resolve() == resolved_parent:
            return category
    return "concept"


def resolve_slug_path(
    cfg: dict[str, Any],
    slug: str,
    category: str | None = None,
    root: Path | None = None,
    for_write: bool = False,
) -> Path:
    """
    Risolve uno `slug` attraverso tutte le directory di `paths.content_dirs`.
    Se il file esiste già, restituisce il Path reale; se non esiste (es. `kb_compile`),
    lo indirizza nella directory corrispondente a `category` (default `paths.concepts`).
    """
    clean_slug = slug[:-3] if slug.endswith(".md") else slug
    clean_slug = Path(clean_slug).name
    filename = f"{clean_slug}.md"

    base = root or get_project_root(cfg)
    content_dirs = get_content_dirs(cfg, base)

    if category and category in content_dirs:
        candidate = content_dirs[category] / filename
        if candidate.is_file() and is_active_kb_document(candidate, category):
            return candidate

    for cat, folder in content_dirs.items():
        candidate = folder / filename
        if candidate.is_file() and is_active_kb_document(candidate, cat):
            return candidate

    if category and category in content_dirs:
        target_dir = content_dirs[category]
        if for_write:
            target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / filename

    concepts_dir = content_dirs.get("concept")
    if concepts_dir is None:
        concepts_rel = cfg.get("paths", {}).get("concepts", "wiki/concepts")
        concepts_dir = Path(concepts_rel)
        if not concepts_dir.is_absolute():
            concepts_dir = base / concepts_dir
    if for_write:
        concepts_dir.mkdir(parents=True, exist_ok=True)
    return concepts_dir / filename


def strip_cjk_sections_if_bare_body(text: str, cfg: dict[str, Any] | None = None) -> str:
    """Rimuove eventuali blocchi `## 中文` e `## 日本語` se `language.bare_body` è True."""
    bare = True
    if isinstance(cfg, dict):
        lang = cfg.get("language", {})
        if isinstance(lang, dict) and "bare_body" in lang:
            bare = bool(lang.get("bare_body"))
    if not bare:
        return text
    return CJK_SECTION_RE.sub("", text).rstrip() + "\n"


@contextlib.contextmanager
def meta_write_lock(cfg: dict[str, Any], root: Path | None = None) -> Iterator[None]:
    """Acquisisce un lock esclusivo (`fcntl.flock`) su `wiki/_meta/.lock`."""
    base = root or get_project_root(cfg)
    meta_rel = cfg.get("paths", {}).get("meta", "wiki/_meta")
    meta_dir = Path(meta_rel)
    if not meta_dir.is_absolute():
        meta_dir = base / meta_dir
    meta_dir.mkdir(parents=True, exist_ok=True)
    lock_file = meta_dir / ".lock"
    with open(lock_file, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def atomic_write_text(
    target_path: Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """Scrive `content` su `target_path` in modo atomico (`tempfile` + `os.replace`)."""
    target_path = Path(target_path)
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
        if os.path.exists(tmp_name):
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
        raise
