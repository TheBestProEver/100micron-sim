"""The decision map (docs/ENGINE_LOGIC.md) and the design record (docs/design/at-limit-sleep/README.md) must name
only functions, checks and tests that exist, so that what they say can always be traced to the code."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "dolle_sim"
DOCS = [ROOT / "docs" / "ENGINE_LOGIC.md", ROOT / "docs" / "design" / "at-limit-sleep" / "README.md"]
MODULES = ("engine", "statics", "loads", "dem", "slaved", "roll1", "flight", "mission", "simulation")


def _src(name):
    return (PKG / f"{name}.py").read_text(encoding="utf-8")


def _defined(src, name):
    return re.search(rf"^\s*(def|class)\s+{re.escape(name)}\b", src, re.M) is not None


def _check_ids():
    ids = set()
    for f in ("verification.py", "verify_full.py"):
        ids |= set(re.findall(r'_res\(\s*"(\w+)"', (PKG / f).read_text(encoding="utf-8")))
    return ids


def _tests():
    names = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        names |= set(re.findall(r"^def (test_\w+)", f.read_text(encoding="utf-8"), re.M))
    return names


def _text():
    return {p.name: p.read_text(encoding="utf-8") for p in DOCS}


def test_named_functions_exist():
    all_src = "\n".join(_src(m) for m in MODULES)
    for doc, text in _text().items():
        for mod, rest in re.findall(rf"`({'|'.join(MODULES)})\.([\w.]+)`", text):
            if rest == "py":                      # a file name, not a function
                continue
            name = rest.split(".")[-1]
            assert _defined(_src(mod), name), f"{doc}: {mod}.{rest} is not defined in dolle_sim/{mod}.py"
        for name in re.findall(r"`(_[a-z]\w*)`", text):
            assert _defined(all_src, name), f"{doc}: {name} is not defined in dolle_sim"
        for cls, meth in re.findall(r"`([A-Z]\w+)\.(\w+)(?:\([^`]*\))?`", text):
            assert _defined(all_src, cls) and _defined(all_src, meth), f"{doc}: {cls}.{meth} is not defined"


def test_named_checks_and_tests_exist():
    ids, tests = _check_ids(), _tests()
    for doc, text in _text().items():
        for c in re.findall(r"`check:(\w+)`", text):
            assert c in ids, f"{doc}: check {c} does not exist"
        for c in re.findall(r"`planned:(\w+)`", text):
            assert c not in ids, f"{doc}: check {c} exists now: write it as `check:{c}`"
        for t in re.findall(r"`(test_\w+)`", text):
            assert t in tests, f"{doc}: test {t} does not exist"
