#!/usr/bin/env python3
"""Sandbox test for glm53-nixl-probe-skip (no GPU, no vllm/torch import).

Builds a fake dist-packages tree, runs the patcher against it via
GLM53_VLLM_DIR (the sandbox anchor convention — never the live image tree),
then execs the patched has_nixl_ep + _nixl_backend_matches_torch in a stub
namespace (module body imports vllm.logger, so the whole file cannot load
outside the image) and asserts the behavior matrix:

  1. patch applies to the stock tree exactly once; idempotent on re-run
  2. anchor drift (pre-patched / modified tree) -> refused, exit 1
  3. backend .so tagged for ANOTHER torch  -> has_nixl_ep() == False and the
     doomed import (_has_module) is NEVER called (no traceback possible)
  4. backend .so tagged for THIS torch     -> falls through to _has_module
  5. no nixl_ep_cu13 package at all        -> False, no import
  6. GLM53_NIXL_PROBE_SKIP=0               -> stock path (_has_module called)
  7. probe errors (unreadable version.py)  -> fail-open to _has_module
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOCK = Path(os.environ.get(
    "GLM53_TEST_STOCK_IMPORT_UTILS",
    "/tmp/nixl-sandbox/import_utils.stock.py"))
MOD = HERE / "patch_nixl_probe_skip.py"


def run_patcher(tree: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GLM53_VLLM_DIR"] = str(tree / "vllm")
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(MOD)], env=env, capture_output=True, text=True)


def make_tree(tmp: Path, *, backend_so: str | None, torch_version: str = "2.14.0a0",
              with_backend_pkg: bool = True, with_torch: bool = True) -> Path:
    dist = tmp / "dist-packages"
    (dist / "vllm" / "utils").mkdir(parents=True)
    shutil.copy(STOCK, dist / "vllm" / "utils" / "import_utils.py")
    if with_backend_pkg:
        pkg = dist / "nixl_ep_cu13"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        if backend_so:
            (pkg / backend_so).write_bytes(b"\0")
    if with_torch:
        (dist / "torch").mkdir()
        (dist / "torch" / "__init__.py").write_text("")
        (dist / "torch" / "version.py").write_text(
            f'__version__ = "{torch_version}"\n')
    return dist


def load_patched_has_nixl_ep(dist: Path, env_extra: dict | None = None):
    """Exec ONLY the two patched functions in a stub namespace."""
    import ast as _ast
    import sys as _sys

    src = (dist / "vllm" / "utils" / "import_utils.py").read_text()
    tree = _ast.parse(src)
    shared: dict = {"os": os, "Path": Path,
                    "importlib": __import__("importlib"),
                    "re": __import__("re"),
                    "_has_module": lambda name: calls.append(name) or True}
    calls: list = []
    shared["_has_module"] = lambda name: calls.append(name) or True
    # sandbox tree must be importable-by-spec for find_spec to see the fakes;
    # the path entry stays until the caller pops it (the returned fn needs it)
    importlib = __import__("importlib")
    importlib.invalidate_caches()
    _sys.path.insert(0, str(dist))
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name in (
                "has_nixl_ep", "_nixl_backend_matches_torch"):
            exec(compile(_ast.Module(body=[node], type_ignores=[]),
                         "<patched>", "exec"), shared)
    return shared["has_nixl_ep"], calls, dist


def main() -> int:
    assert STOCK.exists(), f"stock import_utils missing: {STOCK}"
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="nixl-sandbox-"))
    try:
        # 1. applies once, idempotent
        dist = make_tree(tmp / "t1", backend_so="nixl_ep_cpp_torch216.cpython-312-aarch64-linux-gnu.so")
        r1 = run_patcher(dist)
        check("patch applies", r1.returncode == 0 and "[glm53-nixl-probe-skip] applied" in r1.stdout, r1.stdout.strip()[-80:])
        r2 = run_patcher(dist)
        check("idempotent re-run", r2.returncode == 0 and "already applied" in r2.stdout, r2.stdout.strip()[-80:])
        patched = (dist / "vllm" / "utils" / "import_utils.py").read_text()
        check("single marker", patched.count("[glm53-nixl-probe-skip]") >= 1 and patched.count("def has_nixl_ep") == 1)
        import ast as _ast
        _ast.parse(patched)

        # 3. mismatched backend -> False, no doomed import
        has_nixl_ep, calls, p1 = load_patched_has_nixl_ep(dist)
        check("mismatched .so -> False", has_nixl_ep() is False)
        check("no import attempted on mismatch", calls == [], str(calls))
        sys.path.remove(str(p1))

        # 4. matched backend -> falls through to _has_module
        dist_m = make_tree(tmp / "t4", backend_so="nixl_ep_cpp_torch214.cpython-312-aarch64-linux-gnu.so")
        run_patcher(dist_m)
        has_m, calls_m, p4 = load_patched_has_nixl_ep(dist_m)
        check("matched .so -> stock path", has_m() is True and calls_m == ["nixl_ep"], str(calls_m))
        sys.path.remove(str(p4))

        # 5. no backend package -> False, no import
        dist_n = make_tree(tmp / "t5", backend_so=None, with_backend_pkg=False)
        run_patcher(dist_n)
        has_n, calls_n, p5 = load_patched_has_nixl_ep(dist_n)
        check("no backend pkg -> False", has_n() is False and calls_n == [])
        sys.path.remove(str(p5))

        # 6. kill-switch restores stock
        src6 = (dist / "vllm" / "utils" / "import_utils.py").read_text()
        os.environ["GLM53_NIXL_PROBE_SKIP"] = "0"
        try:
            has_k, calls_k, pk = load_patched_has_nixl_ep(dist)
            check("GLM53_NIXL_PROBE_SKIP=0 -> stock", has_k() is True and calls_k == ["nixl_ep"])
            sys.path.remove(str(pk))
        finally:
            os.environ.pop("GLM53_NIXL_PROBE_SKIP", None)

        # 7. probe error -> fail-open to stock import
        dist_e = make_tree(tmp / "t7", backend_so="nixl_ep_cpp_torch216.cpython-312-aarch64-linux-gnu.so")
        run_patcher(dist_e)
        (dist_e / "torch" / "version.py").chmod(0o000)  # unreadable -> probe fails
        has_e, calls_e, p7 = load_patched_has_nixl_ep(dist_e)
        check("probe error -> fail-open", has_e() is True and calls_e == ["nixl_ep"])
        sys.path.remove(str(p7))
        (dist_e / "torch" / "version.py").chmod(0o644)

        # 2. anchor drift: already-patched tree -> refuse
        r3 = run_patcher(dist)  # dist is patched; expect no-op (covered by r2), drift = edited anchor
        drifted = make_tree(tmp / "t2", backend_so="nixl_ep_cpp_torch216.cpython-312-aarch64-linux-gnu.so")
        s = (drifted / "vllm" / "utils" / "import_utils.py").read_text()
        s = s.replace('return _has_module("nixl_ep")', 'return _has_module("nixl_ep")  # edited')
        (drifted / "vllm" / "utils" / "import_utils.py").write_text(s)
        r4 = run_patcher(drifted)
        check("anchor drift refused", r4.returncode == 1 and "ANCHOR DRIFT" in r4.stderr, r4.stderr.strip()[-80:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
