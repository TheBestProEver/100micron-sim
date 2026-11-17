"""The decision log (docs/DECISIONS.md) must name only things that exist, so that every decision can be traced to the
code: file paths, `module.function` names, `check:<id>` checks, test names and `commit:<hash>` commits written in
code font. It must also be complete in itself: every entry in the index has a section and the other way round, and
every "superseded by" points at an entry that exists."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "DECISIONS.md"
PACKAGES = {"dolle_core": ROOT / "dolle_core", "dolle_sim": ROOT / "dolle_sim",
            "dolle_extras": ROOT / "extras" / "dolle_extras"}
FILE_EXT = (".py", ".md", ".json", ".ipynb", ".html", ".js", ".css", ".toml", ".txt")
ID = r"[A-HO]\d{1,2}"


def _text():
    return DOC.read_text(encoding="utf-8")


def _code_spans(text):
    """Everything written in code font, without the fenced code blocks (python examples)."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.findall(r"`([^`\n]+)`", text)


def _looks_like_path(word):
    word = word.strip("'\",;:()")
    if not word or word.startswith(("http", "-", "<", "commit:", "check:")) or "*" in word or "=" in word:
        return None
    if word.startswith("results/"):                       # output files written by the commands (git-ignored)
        return None
    if "/" in word or word.endswith(FILE_EXT):
        return word
    return None


def test_file_paths_exist():
    missing = []
    for span in _code_spans(_text()):
        for word in span.split():
            p = _looks_like_path(word)
            if p and not (ROOT / p).exists():
                missing.append(p)
    assert not missing, f"paths named in docs/DECISIONS.md that do not exist: {sorted(set(missing))}"


def _module_file(parts):
    """The file of the longest module prefix of parts (a dotted name), and the remaining attribute names."""
    base = PACKAGES[parts[0]]
    for k in range(len(parts), 0, -1):
        sub = base.joinpath(*parts[1:k])
        for f in (sub.with_suffix(".py"), sub / "__init__.py"):
            if k > 1 and f.exists():
                return f, parts[k:]
        if k == 1 and (base / "__init__.py").exists():
            return base / "__init__.py", parts[1:]
    return None, parts


def _defined_at_top(src, name):
    return re.search(rf"^(def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*[:=]", src, re.M) is not None


def _defined_inside(src, name):
    return re.search(rf"^\s+(def\s+{re.escape(name)}\b|{re.escape(name)}\s*[:=])", src, re.M) is not None


def test_module_functions_exist():
    bad = []
    for span in _code_spans(_text()):
        m = re.fullmatch(r"(dolle_core|dolle_sim|dolle_extras)((?:\.\w+)*)", span.strip())
        if not m:
            continue
        parts = span.strip().split(".")
        f, attrs = _module_file(parts)
        if f is None:
            bad.append(span)
            continue
        if not attrs:
            continue
        src = f.read_text(encoding="utf-8")
        if not _defined_at_top(src, attrs[0]):
            if f.name == "__init__.py" and re.search(rf"\bimport\b[^\n]*\b{re.escape(attrs[0])}\b", src):
                continue                                  # re-exported by the package
            bad.append(span)
            continue
        if any(not _defined_inside(src, a) for a in attrs[1:]):
            bad.append(span)
    assert not bad, f"module.function names in docs/DECISIONS.md that are not defined: {sorted(set(bad))}"


def _check_ids():
    ids = set()
    for f in (ROOT / "dolle_sim" / "verification.py", ROOT / "extras" / "dolle_extras" / "verify_full.py"):
        ids |= set(re.findall(r'_res\(\s*"(\w+)"', f.read_text(encoding="utf-8")))
    return ids


def test_named_checks_exist():
    ids = _check_ids()
    assert len(ids) == 45                                  # 26 model checks + 19 engine checks (CHECKS.md)
    named = re.findall(r"`check:(\w+)`", _text())
    assert named, "no check is named"
    missing = sorted(set(c for c in named if c not in ids))
    assert not missing, f"checks named in docs/DECISIONS.md that do not exist: {missing}"


def test_named_tests_exist():
    names = set()
    for f in [*(ROOT / "tests").glob("test_*.py"), *(ROOT / "extras" / "tests").glob("test_*.py")]:
        names |= set(re.findall(r"^def (test_\w+)", f.read_text(encoding="utf-8"), re.M))
    missing = sorted(set(t for t in re.findall(r"`(test_\w+)`", _text()) if t not in names))
    assert not missing, f"tests named in docs/DECISIONS.md that do not exist: {missing}"


def test_named_commits_exist():
    commits = sorted(set(re.findall(r"`commit:([0-9a-f]{7,40})`", _text())))
    assert commits, "no commit is named"
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    missing = [c for c in commits
               if subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=ROOT,
                                 capture_output=True).returncode != 0]
    assert not missing, f"commits named in docs/DECISIONS.md that do not exist: {missing}"


def _sections():
    """Entry IDs with a heading (### X1. ...) and IDs in the rows of the H and O tables."""
    text = _text()
    heads = re.findall(rf"^### ({ID})\. ", text, re.M)
    rows = re.findall(r"^\| ([HO]\d{1,2}) \|", text, re.M)
    return heads, rows


def test_index_and_entries_agree():
    text = _text()
    heads, rows = _sections()
    assert len(heads) == len(set(heads)), "an entry ID is used twice"
    index = text.split("## 2. Index", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(rf"^\| ({ID}) \|", index, re.M)
    assert listed == heads, "the index and the entries differ (order or IDs)"
    assert len(rows) == len(set(rows)), "an H or O ID is used twice"


def test_references_between_entries_exist():
    text = _text()
    heads, rows = _sections()
    known = set(heads) | set(rows)
    for ref in re.findall(rf"superseded(?: in part)? by ({ID})", text):
        assert ref in known, f"'superseded by {ref}': no such entry"
    for ref in re.findall(rf"\((?:see )?({ID})\)", text):
        assert ref in known, f"reference ({ref}): no such entry"


def test_public_text_rule():
    text = re.sub(r"\s+", " ", _text())                    # a title may be wrapped across lines
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), "an e-mail address in docs/DECISIONS.md"
    for m in re.finditer(r"RockSat", text):                # the launch provider's guide, cited by its title only
        assert text.startswith("RockSat User Guide 2026", m.start()), text[m.start() - 40:m.end() + 40]
