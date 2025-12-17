#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/ianreitsma/projects/starcoder")


def run(cmd: list[str] | str, *, check: bool = True) -> int:
    if isinstance(cmd, str):
        p = subprocess.run(cmd, shell=True, cwd=REPO)
    else:
        p = subprocess.run(cmd, cwd=REPO)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def patch_requirements() -> None:
    req = REPO / "requirements.txt"
    txt = req.read_text().splitlines()

    acc_comment = "# Optional: For better performance"
    acc_line = "accelerate>=0.24.0  # Multi-GPU/distributed training support"
    pytest_comment = "# Optional test dependency"
    pytest_line = "pytest>=7.4.0"

    while txt and txt[-1].strip() == "":
        txt.pop()

    has_acc = any(l.strip().startswith("accelerate>=") for l in txt)
    has_pytest = any(l.strip().startswith("pytest>=") for l in txt)

    if not has_acc:
        insert_at = len(txt)
        for i, l in enumerate(txt):
            if l.strip() == pytest_comment or l.strip().startswith("pytest>="):
                insert_at = i
                break
        block = ["", acc_comment, acc_line, ""]
        txt[insert_at:insert_at] = block

    if not has_pytest:
        if txt and txt[-1].strip() != "":
            txt.append("")
        txt += [pytest_comment, pytest_line]

    out: list[str] = []
    for l in txt:
        if l.strip() == "" and (not out or out[-1].strip() == ""):
            continue
        out.append(l)

    req.write_text("\n".join(out) + "\n")


def patch_git_scraper_dynamic() -> None:
    scr = REPO / "scrapers" / "git_scraper_dynamic.py"
    if not scr.exists():
        return

    s = scr.read_text()

    # Force the pygit2 optional-import block into a safe canonical form.
    # This avoids leaving a broken `except ImportError:` with no body.
    s2 = re.sub(
        r"try:\n\s+import pygit2\nexcept ImportError:\n(?:[\s\S]*?)(?:\n\n|\Z)",
        "try:\n    import pygit2\nexcept ImportError:\n    pygit2 = None\n\n",
        s,
        flags=re.M,
    )

    # Also remove any remaining hard exits tied to pygit2 messaging.
    s2 = re.sub(
        r"print\(\"Install: pip install GitPython pygit2 tqdm\"\)\n\s*sys\.exit\(1\)",
        "# pygit2 is optional; GitPython fallback will be used.\npygit2 = None",
        s2,
    )

    # Add fallback at call site if we find the common pattern.
    if "return self._analyze_with_pygit2(repo_path)" in s2:
        s2 = s2.replace(
            "return self._analyze_with_pygit2(repo_path)",
            "# Prefer pygit2 when available; otherwise fall back to GitPython.\n        if pygit2 is None:\n            try:\n                logger.warning(\"pygit2 not available; falling back to GitPython-based analysis\")\n            except Exception:\n                pass\n            return self._analyze_with_gitpython(repo_path)\n        return self._analyze_with_pygit2(repo_path)",
        )

    if s2 != s:
        scr.write_text(s2)


def patch_dataset_builder_enhanced() -> None:
    path = REPO / "dataset_builder_enhanced.py"
    if not path.exists():
        return
    s = path.read_text()

    # Fix common annotation typos introduced by generators in function signatures:
    #   `param List[...]` -> `param: List[...]`
    #   `param Dict[...]` -> `param: Dict[...]`, etc.
    s2 = re.sub(r"\b([A-Za-z_]\w*)\s+(List|Dict|Optional|Tuple)\[", r"\1: \2[", s)

    # Also fix a very specific historical typo that appears in this repo.
    s2 = re.sub(r"\bmeta\s+List\[", "meta: List[", s2)

    if s2 != s:
        path.write_text(s2)


def main() -> None:
    patch_requirements()
    patch_git_scraper_dynamic()
    patch_dataset_builder_enhanced()

    venv_py = REPO / "venv" / "bin" / "python3"
    py = str(venv_py) if venv_py.exists() else sys.executable

    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    run([py, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    run([
        py,
        "-m",
        "py_compile",
        "training/model_trainer_unified.py",
        "device_backend.py",
        "model_trainer_metal_cuda.py",
        "scrapers/git_scraper_dynamic.py",
        "dataset_builder_enhanced.py",
    ])

    run([py, "-m", "unittest", "discover", "-v"], check=True)


if __name__ == "__main__":
    main()
