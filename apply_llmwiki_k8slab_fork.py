#!/usr/bin/env python3
"""
apply_llmwiki_k8slab_fork.py — Patcher idempotente per il fork GitHub di llmwiki (v0.9.1-k8slab).

Scopo:
  Estende un clone del repository upstream `llmwiki==0.9.0` con le 4 funzionalità
  architetturali richieste da `k8s-lab`:
    1. Supporto multi-cartella nativo tramite `paths.content_dirs` in `config.yaml`
       (concepts, entities, patterns, workflows, procedures, plans).
    2. Filtro selettivo per `wiki/plans/*.md`: indicizza solo i piani con
       `status: active | approved | in_progress`, ignorando i piani storici/archiviati.
    3. Normalizzazione trasparente del frontmatter `k8s-lab` in lettura e lint:
       - `type` -> `category`
       - `last_updated` -> `updated`
       - rimozione del prefisso `#` dai tag (es. `"#mcp"` -> `"mcp"`)
       - rimozione opzionale in memoria di eventuali sezioni CJK residue (`## 中文`, `## 日本語`)
         quando il profilo lingua è `bare_body: true`.
    4. Scrittura atomica (`tempfile` + `os.replace`) protetta da file-locking esclusivo
       (`fcntl.flock` su `wiki/_meta/.lock`) per `index.json`, `_index.md` e `log.md`.

Utilizzo:
  python3 apply_llmwiki_k8slab_fork.py /percorso/al/clone/di/llmwiki [--verify]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

K8SLAB_DIRS_MODULE = '''\
"""
llmwiki.k8slab_dirs — Estensione multi-cartella, filtro piani attivi, normalizzazione
frontmatter e file-locking atomico per k8s-lab (v0.9.1-k8slab).
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
FRONTMATTER_RE = re.compile(r"^---\\s*\\n(.*?)\\n---\\s*\\n?(.*)$", re.DOTALL)
CJK_SECTION_RE = re.compile(
    r"\\n##\\s+(?:中文|日本語)\\b.*?(?=\\n##\\s+(?!中文|日本語)|\\Z)",
    re.DOTALL,
)

# Categoria -> cartella di default se `paths.content_dirs` non è esplicitato
DEFAULT_CONTENT_DIRS_ORDER = [
    ("concept", "concepts"),
    ("entity", "entities"),
    ("pattern", "patterns"),
    ("workflow", "workflows"),
    ("procedure", "procedures"),
    ("plan", "plans"),
]


def get_project_root(cfg: dict[str, Any] | None = None) -> Path:
    """Risolve la root del progetto a partire da cfg['_root'] o dalla CWD."""
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
    Se `paths.content_dirs` è definito in `config.yaml`, usa esattamente quelle directory;
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
      - `last_updated` or `created_at` -> `updated` (se `updated` è assente)
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
    (più `paths.outputs` se `include_outputs=True`), filtrando i piani non attivi
    ed evitando duplicati di slug.
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
    Risolve uno `slug` (con o senza `.md`) attraverso tutte le directory di `paths.content_dirs`:
      1. Se il file esiste già in una delle `content_dirs` (e, se è un `plan`, è attivo),
         restituisce il `Path` reale esistente.
      2. Se `for_write=True` o il file non esiste ancora, lo indirizza nella cartella
         corrispondente a `category` (es. `entity` -> `wiki/entities/<slug>.md`),
         con fallback su `paths.concepts` (`wiki/concepts/<slug>.md`).
    """
    clean_slug = slug[:-3] if slug.endswith(".md") else slug
    clean_slug = Path(clean_slug).name
    filename = f"{clean_slug}.md"

    base = root or get_project_root(cfg)
    content_dirs = get_content_dirs(cfg, base)

    # 1. Se category è specificata e il file esiste già lì, ritorna subito
    if category and category in content_dirs:
        candidate = content_dirs[category] / filename
        if candidate.is_file() and is_active_kb_document(candidate, category):
            return candidate

    # 2. Cerca un file esistente in tutte le content_dirs nell'ordine dichiarato
    for cat, folder in content_dirs.items():
        candidate = folder / filename
        if candidate.is_file() and is_active_kb_document(candidate, cat):
            return candidate

    # 3. Destinazione per nuovi file (o file non trovati)
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
    """
    Acquisisce un file-lock esclusivo (`fcntl.flock`) su `wiki/_meta/.lock`
    per proteggere le scritture concorrenti su `index.json`, `_index.md` e `log.md`.
    """
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
    """
    Scrive `content` su `target_path` in modo atomico usando un file temporaneo
    nella stessa directory (stesso filesystem ZFS/APFS) seguito da `os.replace()`.
    """
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
'''

CONFIG_HOOK_MARKER = "# === K8S-LAB FORK EXTENSION (v0.9.1-k8slab) ==="

CONFIG_HOOK_CODE = f"""
{CONFIG_HOOK_MARKER}
from llmwiki.k8slab_dirs import (
    get_content_dirs,
    iter_kb_markdown_files,
    resolve_slug_path,
    normalize_k8slab_frontmatter,
)

_orig_defaults = globals().get("_defaults")
if callable(_orig_defaults):
    def _defaults():
        d = _orig_defaults()
        paths = d.setdefault("paths", {{}})
        paths.setdefault(
            "content_dirs",
            {{
                "concept": paths.get("concepts", "wiki/concepts"),
                "entity": "wiki/entities",
                "pattern": "wiki/patterns",
                "workflow": "wiki/workflows",
                "procedure": "wiki/procedures",
                "plan": "wiki/plans",
            }},
        )
        return d
"""

SEARCH_HOOK_CODE = f"""
{CONFIG_HOOK_MARKER}
from llmwiki.k8slab_dirs import (
    iter_kb_markdown_files,
    detect_category_for_path,
    parse_markdown_frontmatter,
    normalize_k8slab_frontmatter,
)
"""

OPERATIONS_HOOK_CODE = f"""
{CONFIG_HOOK_MARKER}
from llmwiki.k8slab_dirs import (
    get_content_dirs,
    iter_kb_markdown_files,
    detect_category_for_path,
    resolve_slug_path,
    parse_markdown_frontmatter,
    normalize_k8slab_frontmatter,
    meta_write_lock,
    atomic_write_text,
)

# Avvolge kb_rebuild_index, kb_compile, kb_file_answer, kb_ingest con il lock atomico su _meta/.lock
for _mut_fn_name in ("kb_rebuild_index", "kb_compile", "kb_file_answer", "kb_ingest"):
    _orig_mut_fn = globals().get(_mut_fn_name)
    if callable(_orig_mut_fn):
        def _make_locked_fn(_fn):
            def _locked_wrapper(cfg, *args, **kwargs):
                with meta_write_lock(cfg):
                    return _fn(cfg, *args, **kwargs)
            _locked_wrapper.__name__ = _fn.__name__
            _locked_wrapper.__doc__ = _fn.__doc__
            return _locked_wrapper
        globals()[_mut_fn_name] = _make_locked_fn(_orig_mut_fn)
"""

TAXONOMY_HOOK_CODE = f"""
{CONFIG_HOOK_MARKER}
from llmwiki.k8slab_dirs import (
    get_content_dirs,
    iter_kb_markdown_files,
    normalize_k8slab_frontmatter,
)
"""


def patch_search_py(search_path: Path) -> list[str]:
    """Aggiorna llmwiki/search.py per usare iter_kb_markdown_files su tutte le content_dirs."""
    changes: list[str] = []
    text = search_path.read_text(encoding="utf-8")

    # Sostituisce il pattern upstream `list(concepts_dir.glob("*.md")) + list(outputs_dir.glob("*.md"))`
    pattern_combined = re.compile(
        r"list\(\s*concepts_dir\.glob\(\s*['\"]\*\.md['\"]\s*\)\s*\)\s*\+\s*list\(\s*outputs_dir\.glob\(\s*['\"]\*\.md['\"]\s*\)\s*\)"
    )
    if pattern_combined.search(text):
        text = pattern_combined.sub(
            "iter_kb_markdown_files(cfg, include_outputs=True)",
            text,
        )
        changes.append("search.py: sostituito concepts_dir.glob + outputs_dir.glob con iter_kb_markdown_files(cfg, include_outputs=True)")

    # Sostituisce eventuali `concepts_dir.glob("*.md")` residui
    pattern_concepts_glob = re.compile(r"concepts_dir\.glob\(\s*['\"]\*\.md['\"]\s*\)")
    if pattern_concepts_glob.search(text):
        text = pattern_concepts_glob.sub("iter_kb_markdown_files(cfg, include_outputs=False)", text)
        changes.append("search.py: sostituito concepts_dir.glob('*.md') con iter_kb_markdown_files(cfg)")

    if CONFIG_HOOK_MARKER not in text:
        text = SEARCH_HOOK_CODE.lstrip() + "\n" + text
        changes.append("search.py: aggiunto import da llmwiki.k8slab_dirs")

    search_path.write_text(text, encoding="utf-8")
    return changes


def patch_operations_py(ops_path: Path) -> list[str]:
    """Aggiorna llmwiki/operations.py per multi-dir, risoluzione slug, scrittura atomica e lock."""
    changes: list[str] = []
    text = ops_path.read_text(encoding="utf-8")

    # 1. Sostituisce `concepts_dir.glob("*.md")` con `iter_kb_markdown_files(cfg)`
    pattern_glob = re.compile(r"(?:sorted\()?list\(concepts_dir\.glob\(\s*['\"]\*\.md['\"]\s*\)\)|concepts_dir\.glob\(\s*['\"]\*\.md['\"]\s*\)")
    if pattern_glob.search(text):
        text = pattern_glob.sub("iter_kb_markdown_files(cfg)", text)
        changes.append("operations.py: sostituito concepts_dir.glob('*.md') con iter_kb_markdown_files(cfg)")

    # 2. Sostituisce `Path(cfg["paths"]["concepts"]) / f"{slug}.md"` e `concepts_dir / f"{slug}.md"`
    pattern_slug_direct = re.compile(
        r"(?:Path\(\s*cfg\[['\"]paths['\"]\]\[['\"]concepts['\"]\]\s*\)|concepts_dir)\s*/\s*f['\"]\{slug\}\.md['\"]"
    )
    if pattern_slug_direct.search(text):
        text = pattern_slug_direct.sub("resolve_slug_path(cfg, slug)", text)
        changes.append("operations.py: sostituito path fisso concepts/{slug}.md con resolve_slug_path(cfg, slug)")

    # 3. Sostituisce scritture su index_path / index_md_path con atomic_write_text
    pattern_write = re.compile(r"(\b(?:index_path|index_json_path|index_md_path|out_path|article_path|target_path))\s*\.\s*write_text\(([^)]+)\)")
    if pattern_write.search(text):
        text = pattern_write.sub(r"atomic_write_text(\1, \2)", text)
        changes.append("operations.py: convertite le scritture file in atomic_write_text()")

    if CONFIG_HOOK_MARKER not in text:
        text = text.rstrip() + "\n\n" + OPERATIONS_HOOK_CODE.lstrip()
        changes.append("operations.py: registrati helper k8slab_dirs e wrapper meta_write_lock")

    ops_path.write_text(text, encoding="utf-8")
    return changes


def patch_taxonomy_py(tax_path: Path) -> list[str]:
    """Aggiorna llmwiki/taxonomy.py per scansionare tutte le content_dirs."""
    if not tax_path.is_file():
        return []
    changes: list[str] = []
    text = tax_path.read_text(encoding="utf-8")

    pattern_glob = re.compile(r"concepts_dir\.glob\(\s*['\"]\*\.md['\"]\s*\)")
    if pattern_glob.search(text):
        text = pattern_glob.sub("iter_kb_markdown_files(cfg)", text)
        changes.append("taxonomy.py: sostituito concepts_dir.glob('*.md') con iter_kb_markdown_files(cfg)")

    if CONFIG_HOOK_MARKER not in text:
        text = TAXONOMY_HOOK_CODE.lstrip() + "\n" + text
        changes.append("taxonomy.py: aggiunto import da llmwiki.k8slab_dirs")

    tax_path.write_text(text, encoding="utf-8")
    return changes


def patch_config_py(cfg_path: Path) -> list[str]:
    """Aggiorna llmwiki/config.py con il supporto a paths.content_dirs."""
    changes: list[str] = []
    text = cfg_path.read_text(encoding="utf-8")
    if CONFIG_HOOK_MARKER not in text:
        text = text.rstrip() + "\n\n" + CONFIG_HOOK_CODE.lstrip()
        cfg_path.write_text(text, encoding="utf-8")
        changes.append("config.py: aggiunto supporto paths.content_dirs e re-export helper k8slab_dirs")
    return changes


def apply_fork_patches(repo_dir: Path) -> list[str]:
    pkg_dir = repo_dir / "llmwiki"
    if not pkg_dir.is_dir():
        raise FileNotFoundError(
            f"Directory del pacchetto 'llmwiki/' non trovata in {repo_dir}. "
            "Assicurati di passare la root del repository forkato di llmwiki."
        )

    applied: list[str] = []

    # 1. Scrive il modulo llmwiki/k8slab_dirs.py
    ext_path = pkg_dir / "k8slab_dirs.py"
    standalone_src = repo_dir / "k8slab_dirs.py"
    if standalone_src.is_file():
        ext_path.write_text(standalone_src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        ext_path.write_text(K8SLAB_DIRS_MODULE, encoding="utf-8")
    applied.append(f"Creato/aggiornato {ext_path.relative_to(repo_dir)}")

    # 2. Patch config.py, search.py, operations.py, taxonomy.py
    if (pkg_dir / "config.py").is_file():
        applied.extend(patch_config_py(pkg_dir / "config.py"))
    if (pkg_dir / "search.py").is_file():
        applied.extend(patch_search_py(pkg_dir / "search.py"))
    if (pkg_dir / "operations.py").is_file():
        applied.extend(patch_operations_py(pkg_dir / "operations.py"))
    if (pkg_dir / "taxonomy.py").is_file():
        applied.extend(patch_taxonomy_py(pkg_dir / "taxonomy.py"))

    return applied


def run_self_test(repo_dir: Path) -> None:
    """Esegue un test funzionale in una directory temporanea simulando l'albero k8s-lab."""
    import tempfile

    sys.path.insert(0, str(repo_dir))
    from llmwiki.k8slab_dirs import (
        atomic_write_text,
        iter_kb_markdown_files,
        meta_write_lock,
        normalize_k8slab_frontmatter,
        parse_markdown_frontmatter,
        resolve_slug_path,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = {
            "_root": str(root),
            "paths": {
                "concepts": "wiki/concepts",
                "content_dirs": {
                    "concept": "wiki/concepts",
                    "entity": "wiki/entities",
                    "pattern": "wiki/patterns",
                    "workflow": "wiki/workflows",
                    "procedure": "wiki/procedures",
                    "plan": "wiki/plans",
                },
                "outputs": "wiki/outputs",
                "meta": "wiki/_meta",
            },
            "language": {"profile": "single", "bare_body": True},
        }

        # Crea documenti campione per ogni categoria
        (root / "wiki/entities").mkdir(parents=True)
        (root / "wiki/patterns").mkdir(parents=True)
        (root / "wiki/plans").mkdir(parents=True)
        (root / "wiki/concepts").mkdir(parents=True)

        atomic_write_text(
            root / "wiki/entities/truenas.md",
            "---\ntitle: TrueNAS\ntype: entity\nlast_updated: 2026-10-27\ntags:\n  - '#storage'\n---\nNodo ZFS.\n",
        )
        atomic_write_text(
            root / "wiki/patterns/mcp-registry-package-pattern.md",
            "---\ntitle: MCP Pattern\ntype: pattern\ntags:\n  - '#mcp'\n---\nLink a [[truenas]].\n",
        )
        atomic_write_text(
            root / "wiki/plans/active-plan.md",
            "---\ntitle: Piano Attivo\nstatus: active\n---\nPiano attivo.\n",
        )
        atomic_write_text(
            root / "wiki/plans/old-plan.md",
            "---\ntitle: Piano Vecchio\nstatus: completed\n---\nPiano archiviato.\n",
        )

        files = iter_kb_markdown_files(cfg, root)
        stems = {f.stem for f in files}
        assert stems == {"truenas", "mcp-registry-package-pattern", "active-plan"}, (
            f"Filtro multi-cartella o piani attivi errato: {stems}"
        )

        resolved_entity = resolve_slug_path(cfg, "truenas", root=root)
        assert resolved_entity == root / "wiki/entities/truenas.md"

        new_pattern_target = resolve_slug_path(
            cfg, "new-pattern", category="pattern", root=root, for_write=True
        )
        assert new_pattern_target == root / "wiki/patterns/new-pattern.md"

        meta, _ = parse_markdown_frontmatter((root / "wiki/entities/truenas.md").read_text())
        norm = normalize_k8slab_frontmatter(meta, default_category="entity", slug="truenas")
        assert norm["category"] == "entity"
        assert norm["updated"] == "2026-10-27"
        assert norm["tags"] == ["storage"]

        with meta_write_lock(cfg, root):
            assert (root / "wiki/_meta/.lock").is_file()

    print("SELF-TEST OK: tutte le asserzioni multi-cartella, filtro piani, frontmatter e lock superate!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Applica la patch k8s-lab (v0.9.1-k8slab) al clone del repository llmwiki."
    )
    parser.add_argument(
        "repo_dir",
        nargs="?",
        default=".",
        help="Percorso alla root del repository forkato di llmwiki (default: directory corrente)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Esegue il self-test funzionale subito dopo l'applicazione della patch",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo_dir).resolve()
    applied = apply_fork_patches(repo_path)
    for item in applied:
        print(f"[+] {item}")

    if args.verify:
        run_self_test(repo_path)


if __name__ == "__main__":
    main()
