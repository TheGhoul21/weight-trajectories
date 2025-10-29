#!/usr/bin/env python3
"""
Relativize Markdown links under docs/ so they work on both GitHub and Docsify.

Rules:
- Keep external (http/https/mailto) and anchor-only links unchanged.
- For internal doc links, compute a relative path from the current file to the
  target .md file and include the .md extension (safer on GitHub). Preserve anchors.
- Only process files under docs/; leave code/other paths alone.

Heuristics:
- If link path starts with './' or '../', try resolve relative to current file dir.
- Else, try resolve relative to docs root.
- If the path has no extension, try appending '.md' and resolve.
- If the path has '.md' already, resolve as-is.
- If the resolved file does not exist, leave the link unchanged.

This script updates files in place and prints the files it modified.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

DOCS_ROOT = Path('docs').resolve()

LINK_RE = re.compile(r'(!?\[([^\]]+)\]\(([^)]+)\))')

def split_url(url: str) -> Tuple[str, str]:
    """Split URL into (path, anchor) where anchor includes leading '#' if present."""
    if '#' in url:
        path, anchor = url.split('#', 1)
        return path, '#' + anchor
    return url, ''

def should_skip(url: str) -> bool:
    url = url.strip()
    if not url:
        return True
    low = url.lower()
    return (
        low.startswith('http://') or low.startswith('https://') or low.startswith('mailto:')
        or low.startswith('#') or low.startswith('data:')
    )

def try_candidates(base_dir: Path, path_part: str) -> Path | None:
    candidates = []
    # As-is
    candidates.append((base_dir / path_part))
    # Add .md if missing extension
    if '.' not in Path(path_part).name:
        candidates.append(base_dir / (path_part + '.md'))
    # If path starts with '/', treat as absolute from docs root
    if path_part.startswith('/'):
        pp = path_part.lstrip('/')
        candidates.append(DOCS_ROOT / pp)
        if '.' not in Path(pp).name:
            candidates.append(DOCS_ROOT / (pp + '.md'))
    # If path starts with known top-level within docs
    for prefix in ('manual/', 'scientific/', 'reference/', 'manual', 'scientific', 'reference'):
        if path_part.startswith(prefix):
            pp = path_part
            if pp.startswith('/'):
                pp = pp.lstrip('/')
            candidates.append(DOCS_ROOT / pp)
            if '.' not in Path(pp).name:
                candidates.append(DOCS_ROOT / (pp + '.md'))
            break
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    return None

def relativize_links(md_path: Path) -> bool:
    text = md_path.read_text(encoding='utf-8')
    changed = False
    base_dir = md_path.parent.resolve()

    def replace(m: re.Match) -> str:
        nonlocal changed
        full, label, url = m.group(1), m.group(2), m.group(3)
        # Preserve images and links; treat both
        path_part, anchor = split_url(url)
        if should_skip(path_part):
            return full

        # Already relative with ../ or ./ → check extension; if missing, try add .md if file exists
        resolved: Path | None = None
        if path_part.startswith('./') or path_part.startswith('../') or not ('/' in path_part.split('/')[0]):
            # Try relative to current file first
            rel_target = try_candidates(base_dir, path_part)
            if rel_target is not None:
                resolved = rel_target
            else:
                # Try docs root
                rel_target = try_candidates(DOCS_ROOT, path_part)
                resolved = rel_target
        else:
            # Try docs root, then relative
            rel_target = try_candidates(DOCS_ROOT, path_part)
            if rel_target is None:
                rel_target = try_candidates(base_dir, path_part)
            resolved = rel_target

        if resolved is None:
            return full  # leave unchanged

        # Compute relative path and ensure .md extension preserved
        rel = os.path.relpath(resolved, base_dir)
        # Normalize path separators
        rel = rel.replace('\\', '/')
        new_url = rel + anchor
        if new_url != url:
            changed = True
            prefix = '!' if full.startswith('!') else ''
            return f"{prefix}[{label}]({new_url})"
        return full

    new_text = LINK_RE.sub(replace, text)
    if changed:
        md_path.write_text(new_text, encoding='utf-8')
    return changed


def main() -> None:
    modified = []
    for md_path in DOCS_ROOT.rglob('*.md'):
        if relativize_links(md_path):
            modified.append(str(md_path))
    if modified:
        print('Modified:')
        for p in modified:
            print('  ', p)
    else:
        print('No changes needed.')


if __name__ == '__main__':
    main()

