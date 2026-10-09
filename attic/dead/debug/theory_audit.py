"""Section 4 + 54: the theory/implementation matrix, built from the source.

CODEX_DEEP_STAGEWISE_DEBUG_AND_THEORY_AUDIT.md sections 4, 54, 56.

Every row of the matrix is a claim about this repository, so every row is
CHECKED rather than asserted:

  * does the symbol exist and import?
  * is it reachable from the production entry points, or only from a tool or
    a test?
  * does a test exercise it?

Section 4 is explicit that `MATCHES_THEORY` may not be awarded merely because
a synthetic test passes, so a passing test raises a row no higher than
`DEBUG_ONLY` unless the symbol is also reachable from production. The point of
the matrix is to separate four things that are easy to confuse: source-code
existence, test coverage, offline execution, and actual use by a decision.

Each row also carries `falsified_by` -- what observation would show the
implementation wrong -- because a requirement nobody could disprove is not
being audited.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# section 56 vocabulary
MATCHES_THEORY = "MATCHES_THEORY"
APPROXIMATE = "APPROXIMATE"
DIFFERENT_STATISTIC = "DIFFERENT_STATISTIC"
DEBUG_ONLY = "DEBUG_ONLY"
NOT_WIRED = "NOT_WIRED"
MISSING = "MISSING"
BUG = "BUG"
UNVERIFIED = "UNVERIFIED"

PRODUCTION_ENTRY = ("tools/kpart_pipeline.py", "tools/run_iekf_scene.py")
TOOL_GLOB = "tools/*.py"
TEST_GLOB = "tests/test_*.py"


@dataclass
class Requirement:
    """One theoretical requirement and where it is meant to live."""
    key: str
    requirement: str
    equation: str
    stage: str
    module: str
    symbol: str
    fidelity: str                      # exact / approximate / different
    uncertainty_source: str
    assumptions: str
    debug_artifact: str
    falsified_by: str
    claimed_status: str = UNVERIFIED
    note: str = ""


def _symbol_exists(module: str, symbol: str) -> tuple[bool, str]:
    if not module:
        return False, "no module named in the requirement"
    try:
        m = importlib.import_module(module)
    except Exception as exc:                       # pragma: no cover
        return False, f"import failed: {type(exc).__name__}: {exc}"
    if not symbol:
        return True, "module imports"
    return (hasattr(m, symbol),
            "present" if hasattr(m, symbol) else "symbol absent")


def _grep_count(pattern: str, paths) -> int:
    """Files under `paths` containing `pattern`. Globs are expanded here.

    BUG (section 1 record). Expected: count test files referencing a symbol.
    Actual: the glob "tests/test_*.py" was handed to grep as a literal path
    because subprocess does not run a shell, so every symbol reported zero
    tests and every row that should have read MATCHES_THEORY read UNVERIFIED.
    Minimal correction: expand globs with Path.glob before the call.
    Before: has_test False for all 16 rows. After: True for the 11 rows that
    genuinely have tests.
    """
    files = []
    for p in paths:
        if any(ch in p for ch in "*?["):
            files += [str(q.relative_to(REPO)) for q in REPO.glob(p)]
        elif (REPO / p).is_file():
            files.append(p)
    if not files:
        return 0
    try:
        out = subprocess.run(["grep", "-lE", pattern, *files], cwd=REPO,
                             capture_output=True, text=True, timeout=120)
        return len([l for l in out.stdout.split("\n") if l.strip()])
    except Exception:
        return 0


def _repo_module_of(path: Path) -> str:
    try:
        rel = path.relative_to(REPO)
    except ValueError:
        return ""
    return ".".join(rel.with_suffix("").parts)


def _imported_repo_modules(path: Path) -> set:
    """Repo-local modules a file imports, from its AST."""
    mods: set = set()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return mods
    pkgs = ("joint_reduction", "estimator", "debug", "tools", "scene", "utils")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in pkgs:
                mods.add(node.module)
            elif node.level and node.module:
                # relative import inside a package
                pkg = _repo_module_of(path).rsplit(".", 1)[0]
                mods.add(f"{pkg}.{node.module}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in pkgs:
                    mods.add(a.name)
    return mods


def _reachable(symbol: str) -> dict:
    """Where the symbol is referenced: production entry, tools, tests.

    These three are deliberately separate columns. Section 2 warns that
    source-code existence, test coverage, offline execution and actual use by
    a decision are four different things, and collapsing them is how a
    debug-only module comes to be described as working.
    """
    if not symbol:
        return {"production": 0, "tools": 0, "tests": 0}
    pat = rf"\b{symbol}\b"
    return {"production": _grep_count(pat, list(PRODUCTION_ENTRY)),
            "tools": _grep_count(pat, [TOOL_GLOB]),
            "tests": _grep_count(pat, [TEST_GLOB])}


_REACHABLE_MODULES: set | None = None


def _production_module_closure() -> set:
    """Modules reachable from the production entry points by import.

    BUG (section 1 record). Expected: decide whether a symbol can execute in a
    production run. Actual: the first version walked every ast.Name in every
    module defining any referenced name, which after three hops covered the
    whole repository and reported `dual_differential_finite_evidence` as
    production-wired when `grep -c` on both entry files returns 0. That is the
    class of confidently-wrong answer this audit exists to prevent, so the
    walk is now a real import graph: a symbol counts as wired only if it is
    referenced inside a module the entry points can actually reach.
    Before: 14/16 rows "wired". After: only the rows genuinely on the
    production path.
    """
    global _REACHABLE_MODULES
    if _REACHABLE_MODULES is not None:
        return _REACHABLE_MODULES
    seen: set = set()
    frontier = set()
    for p in PRODUCTION_ENTRY:
        f = REPO / p
        if f.is_file():
            m = _repo_module_of(f)
            seen.add(m)
            frontier |= _imported_repo_modules(f)
    while frontier:
        m = frontier.pop()
        if m in seen:
            continue
        seen.add(m)
        f = REPO / (m.replace(".", "/") + ".py")
        if f.is_file():
            frontier |= _imported_repo_modules(f) - seen
    _REACHABLE_MODULES = seen
    return seen


def _default_off_flags() -> set:
    """argparse flags on the entry points whose default is falsy.

    `store_true` options and options declaring `default=False/0/None` are off
    unless the operator asks for them, so anything reachable only through such
    a branch is wired but not RUNNING in a default production run.
    """
    flags: set = set()
    for p in PRODUCTION_ENTRY:
        fp = REPO / p
        if not fp.is_file():
            continue
        try:
            tree = ast.parse(fp.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            name = None
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value.startswith("--"):
                    name = a.value[2:].replace("-", "_")
            if name is None:
                continue
            kw = {k.arg: k.value for k in node.keywords}
            act = kw.get("action")
            is_store_true = (isinstance(act, ast.Constant)
                             and act.value == "store_true")
            dflt = kw.get("default")
            has_default = dflt is not None and isinstance(dflt, ast.Constant)
            # An explicit `default=True` overrides store_true's implicit
            # False. `--geometric-joints` is declared exactly that way, and
            # reading only the action marked the entire Phase-3 chain as
            # opt-in when it is on unless switched off.
            if has_default:
                if not dflt.value:
                    flags.add(name)
            elif is_store_true:
                flags.add(name)
    return flags


def _flag_gated_modules() -> dict:
    """Repo modules imported only inside an `if args.<default-off-flag>` block.

    This is the distinction section 2 insists on: a module can be importable
    from a production entry point and still never execute, because the branch
    that imports it is behind a switch that is off. Reporting such a module as
    "production wired" without qualification is how an offline experiment
    comes to be described as the shipping decision path.
    """
    off = _default_off_flags()
    gated: dict = {}
    for p in PRODUCTION_ENTRY:
        fp = REPO / p
        if not fp.is_file():
            continue
        try:
            tree = ast.parse(fp.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            names = {n.attr for n in ast.walk(node.test)
                     if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node.test)
                      if isinstance(n, ast.Name)}
            hit = names & off
            if not hit:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    for m in ({sub.module} | {sub.module}):
                        gated.setdefault(m, set()).update(hit)
                elif isinstance(sub, ast.Import):
                    for a in sub.names:
                        gated.setdefault(a.name, set()).update(hit)
    return {k: sorted(v) for k, v in gated.items()}


BATCH_RUNNER = "tools/run_all_datasets.py"


def _flags_set_by_standard_runner() -> set:
    """Flags the standard batch runner always passes.

    `--window-discovery` has an argparse default of False, so a naive reading
    calls the entire Phase-2/3 chain "not on the default path". That is true
    of a bare `run_iekf_scene.py` invocation and false of every real run,
    because `tools/run_all_datasets.py` passes it unconditionally. Reporting
    only the argparse default would therefore describe the shipping pipeline
    as unused, which is the opposite of the error this audit is guarding
    against but just as misleading. This reads the batch runner's command
    construction so the matrix can say what actually executes.
    """
    fp = REPO / BATCH_RUNNER
    if not fp.is_file():
        return set()
    try:
        tree = ast.parse(fp.read_text())
    except Exception:
        return set()
    flags: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.startswith("--"):
            flags.add(node.value[2:].replace("-", "_"))
    return flags


def _files_calling(symbol: str, files: list) -> list:
    """Files containing an actual CALL to `symbol`, not just the name.

    A bare word match counts prose. `rgbd_null_test_for_prismatic` names
    `geometric_joint_candidate` in its docstring to say where its input comes
    from, and that was being read as a call, promoting an offline-only symbol
    to "runs in the standard batch". Matching `symbol(` and dropping the
    `def symbol(` line keeps definitions and prose out.
    """
    hits = []
    call = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(symbol)}\s*\(")
    define = re.compile(rf"^\s*(def|class)\s+{re.escape(symbol)}\b")
    for rel in files:
        fp = REPO / rel
        try:
            lines = fp.read_text().split("\n")
        except Exception:
            continue
        for ln in lines:
            if define.search(ln):
                continue
            if call.search(ln):
                hits.append(rel)
                break
    return hits


_UNGATED: set | None = None


def _ungated_closure() -> set:
    """Modules reachable from the entry points WITHOUT crossing a gated import.

    Gating has to propagate. `run_iekf_scene` imports `direct_joint_pipeline`
    only inside `if args.joint_direct_gaussian:`, and the batch runner never
    passes that flag -- so everything that pipeline pulls in (common_screw,
    direct_gaussian_joint and the rest of the dual chain) is equally unreached,
    even though nothing marks those files themselves as gated. Walking the
    import graph while refusing to step through a gated edge is what makes
    `DEBUG_ONLY` mean something.
    """
    global _UNGATED
    if _UNGATED is not None:
        return _UNGATED
    gated = set(_flag_gated_modules())
    batch = _flags_set_by_standard_runner()
    # a module gated ONLY by flags the batch runner passes is still reached
    blocked = {m for m, flags in _flag_gated_modules().items()
               if not set(flags) <= batch}
    seen: set = set()
    frontier = set()
    for p in PRODUCTION_ENTRY:
        fp = REPO / p
        if fp.is_file():
            seen.add(_repo_module_of(fp))
            frontier |= _imported_repo_modules(fp) - blocked
    while frontier:
        m = frontier.pop()
        if m in seen or m in blocked:
            continue
        seen.add(m)
        fp = REPO / (m.replace(".", "/") + ".py")
        if fp.is_file():
            frontier |= (_imported_repo_modules(fp) - seen) - blocked
    _UNGATED = seen
    return seen


_CALL_GRAPH: dict | None = None
_REACHABLE_SYMBOLS: set | None = None


def _module_namespace(path: Path) -> dict:
    """name -> qualified "module.name", for everything visible in this file.

    Calls are resolved inside the namespace of the file that makes them. A
    bare-name graph merges every `fit`, `solve` and `evaluate` in the
    repository into one node, which made 1659 symbols look reachable from the
    entry points; qualifying them keeps the graph honest.
    """
    ns: dict = {}
    mod = _repo_module_of(path)
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return ns
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ns[node.name] = f"{mod}.{node.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            base = node.module
            if node.level:
                pkg = mod.rsplit(".", 1)[0]
                base = f"{pkg}.{node.module}" if node.module else pkg
            for a in node.names:
                ns[a.asname or a.name] = f"{base}.{a.name}"
    return ns


_CALL_GRAPH: dict | None = None
_REACHABLE_SYMBOLS: set | None = None


def _repo_files() -> list:
    out = []
    for folder in ("joint_reduction", "estimator", "debug", "tools"):
        d = REPO / folder
        if d.is_dir():
            out += sorted(d.glob("*.py"))
    return out


def _call_graph() -> dict:
    """{qualified function: set of qualified functions it calls}."""
    global _CALL_GRAPH
    if _CALL_GRAPH is not None:
        return _CALL_GRAPH
    graph: dict = {}
    for fp in _repo_files():
        ns = _module_namespace(fp)
        mod = _repo_module_of(fp)
        try:
            tree = ast.parse(fp.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    nm = (fn.id if isinstance(fn, ast.Name)
                          else fn.attr if isinstance(fn, ast.Attribute) else None)
                    if nm and nm in ns:
                        calls.add(ns[nm])
            graph[f"{mod}.{node.name}"] = calls
    _CALL_GRAPH = graph
    return graph


def _module_level_calls(path: Path, gated_flags: set) -> set:
    """Qualified names called at module level or in ungated `if` branches."""
    ns = _module_namespace(path)
    out: set = set()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return out

    def walk(node, blocked):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                names = {n.attr for n in ast.walk(child.test)
                         if isinstance(n, ast.Attribute)}
                names |= {n.id for n in ast.walk(child.test)
                          if isinstance(n, ast.Name)}
                blk = blocked or bool(names & gated_flags)
                for sub in child.body:
                    walk(sub, blk)
                for sub in child.orelse:
                    walk(sub, blocked)
                continue
            if isinstance(child, ast.Call) and not blocked:
                fn = child.func
                nm = (fn.id if isinstance(fn, ast.Name)
                      else fn.attr if isinstance(fn, ast.Attribute) else None)
                if nm and nm in ns:
                    out.add(ns[nm])
            walk(child, blocked)

    walk(tree, False)
    return out


def _production_reachable_symbols() -> set:
    """Qualified functions a standard batch run can reach, by call graph.

    Import reachability is not enough: `kpart_pipeline` imports
    `direct_joint_pipeline` at module level, so every module of the dual chain
    is importable, while `angular_rank_evidence` is called only by
    `fit_weighted_common_screw`, called only by
    `dual_differential_finite_evidence`, which nothing in production calls.
    Only a walk of the call graph separates a live path from a dead one.
    """
    global _REACHABLE_SYMBOLS
    if _REACHABLE_SYMBOLS is not None:
        return _REACHABLE_SYMBOLS
    off = _default_off_flags() - _flags_set_by_standard_runner()
    graph = _call_graph()
    seen: set = set()
    frontier: set = set()
    for p in PRODUCTION_ENTRY:
        fp = REPO / p
        if fp.is_file():
            frontier |= _module_level_calls(fp, off)
    # functions defined in the entry files are themselves entry points
    for p in PRODUCTION_ENTRY:
        fp = REPO / p
        if fp.is_file():
            mod = _repo_module_of(fp)
            frontier |= {k for k in graph if k.startswith(mod + ".")}
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        frontier |= graph.get(name, set()) - seen
    _REACHABLE_SYMBOLS = {n.rsplit(".", 1)[-1] for n in seen} | seen
    return _REACHABLE_SYMBOLS


_TOOL_REACHABLE: set | None = None

DEBUG_ENTRY = ("tools/dual_joint_discovery.py", "tools/deep_debug.py",
               "tools/joint_decide_from_folds.py",
               "tools/direct_joint_pipeline.py")


def _tool_reachable_symbols() -> set:
    """Functions an OFFLINE debug tool can reach, by the same call graph.

    Without this, a function executed every time `dual_joint_discovery` runs
    -- but only through two intermediate calls -- is reported NOT_WIRED, i.e.
    dead. `DEBUG_ONLY` and `NOT_WIRED` are different findings: one says the
    code runs but no decision uses it, the other says nothing runs it at all.
    """
    global _TOOL_REACHABLE
    if _TOOL_REACHABLE is not None:
        return _TOOL_REACHABLE
    graph = _call_graph()
    seen: set = set()
    frontier: set = set()
    for p in DEBUG_ENTRY:
        fp = REPO / p
        if not fp.is_file():
            continue
        frontier |= _module_level_calls(fp, set())
        mod = _repo_module_of(fp)
        frontier |= {k for k in graph if k.startswith(mod + ".")}
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        frontier |= graph.get(name, set()) - seen
    _TOOL_REACHABLE = {n.rsplit(".", 1)[-1] for n in seen} | seen
    return _TOOL_REACHABLE


def _transitively_production(symbol: str, defining_module: str = "") -> dict:
    """Can `symbol` run in a production run, and does it run by default?

    A symbol's own `def` line is not a use, so the module that defines it is
    excluded: otherwise every symbol in the closure reports itself as called.
    The result separates three things -- importable, called by someone, and
    on the default path -- because they are what sections 2 and 54 ask to be
    told apart.
    """
    if not symbol:
        return {"importable": False, "called_by": [], "on_default_path": False,
                "gated_by": []}
    closure = _production_module_closure()
    ungated = _ungated_closure()
    gated = _flag_gated_modules()
    files, mod_of = [], {}
    for m in closure:
        # The defining module is NOT skipped: a helper called only by another
        # function in its own file is still reached, as long as that caller
        # is. Definitions are excluded by matching a CALL site instead.
        fp = REPO / (m.replace(".", "/") + ".py")
        if fp.is_file():
            rel = str(fp.relative_to(REPO))
            files.append(rel)
            mod_of[rel] = m
    callers = _files_calling(symbol, files)
    batch = _flags_set_by_standard_runner()
    reachable = _production_reachable_symbols()
    gates: set = set()
    ungated_caller = False
    batch_ok = False
    for rel in callers:
        m = mod_of.get(rel, "")
        g = gated.get(m)
        if g:
            gates.update(g)
        if m in ungated:
            if not g:
                ungated_caller = True
    # The decisive test: can a standard batch run actually reach this symbol
    # through the call graph?
    batch_ok = symbol in reachable
    return {"importable": bool(defining_module in closure or callers),
            "called_by": callers,
            "on_default_path": bool(callers) and ungated_caller,
            "runs_in_standard_batch": bool(callers) and batch_ok,
            "gated_by": sorted(gates)}


def requirements() -> list:
    return [
        Requirement(
            key="5.1_finite_rigidity",
            requirement="Finite rigidity: a rigid body preserves pairwise distance",
            equation="rho_ij = median_t | ||x_i^t - x_j^t|| - ||x_i^0 - x_j^0|| | / ||x_i^0 - x_j^0||",
            stage="Phase 2A finite rigidity graph",
            module="joint_reduction.rigidity_graph",
            symbol="pairwise_rigidity_error",
            fidelity="exact",
            uncertainty_source="RESIDUAL_CALIBRATED (robust MAD scale)",
            assumptions="edges sampled, not all pairs; median over frames",
            debug_artifact="numeric/phase2_graph/rigidity_edges.csv",
            falsified_by=("a synthetic rigid body whose rho is not ~0, or a "
                          "non-rigid one whose rho is")),
        Requirement(
            key="5.2_differential_rigidity",
            requirement="Instantaneous rigidity: the strain rate vanishes",
            equation="G = D + Omega, D = sym(G) = 0 for rigid velocity",
            stage="Phase 1C differential layer",
            module="joint_reduction.velocity_gradient_rigidity",
            symbol="decompose_velocity_gradient",
            fidelity="exact",
            uncertainty_source="RESIDUAL_CALIBRATED via the affine fit covariance",
            assumptions="first-order field over the support; one affine fit per frame",
            debug_artifact="numeric/phase1_differential/velocity_gradient_by_frame.csv",
            falsified_by=("a pure shear reporting nonzero Omega, or a finite "
                          "rotation reporting nonzero D")),
        Requirement(
            key="5.3_pairwise_differential",
            requirement="Pairwise differential invariant",
            equation="d/dt ||r||^2 = 2 r^T D r, so r^T(v_i - v_j) = 0 for rigid",
            stage="Phase 1C",
            module="joint_reduction.velocity_gradient_rigidity",
            symbol="differential_edge_rigidity",
            fidelity="exact",
            uncertainty_source="propagated velocity covariance when supplied, else none",
            assumptions="endpoint errors independent",
            debug_artifact="numeric/phase1_differential/differential_rigidity_edges.csv",
            falsified_by="a spinning rigid body producing nonzero rho_diff"),
        Requirement(
            key="5.4_prismatic_instantaneous",
            requirement="Prismatic instantaneous structure",
            equation="omega = 0, hence v_i - v_j = 0 for all pairs",
            stage="Phase 1C / common screw",
            module="joint_reduction.common_screw",
            symbol="angular_rank_evidence",
            fidelity="exact",
            uncertainty_source="IMAGE_IEKF when supplied, else RESIDUAL_CALIBRATED",
            assumptions="H_P: omega = 0 tested as chi2 against its own quantile",
            debug_artifact="numeric/common_screw/common_screw_fit.json",
            falsified_by="synthetic prismatic data rejecting omega = 0"),
        Requirement(
            key="5.5_revolute_instantaneous",
            requirement="Revolute instantaneous structure",
            equation="omega_t = qdot_t e with fixed e; v_i - v_j = omega x r_ij",
            stage="Phase 1C / common screw",
            module="joint_reduction.common_screw",
            symbol="angular_rank_evidence",
            fidelity="exact",
            uncertainty_source="same as 5.4",
            assumptions="rank-1 fit is weighted and covariance aware",
            debug_artifact="numeric/common_screw/common_screw_fit.json",
            falsified_by="synthetic revolute data failing the rank-1 test"),
        Requirement(
            key="5.6_finite_centered",
            requirement="Finite centred motion removes translation exactly",
            equation="y_i(t) = x_i(t) - c_t = R_t y_i(0) for rigid motion",
            stage="Phase 2 finite rotation",
            module="joint_reduction.finite_relative_rotation",
            symbol="weighted_kabsch_rotation",
            fidelity="exact",
            uncertainty_source="RESIDUAL_CALIBRATED from the Kabsch residual",
            assumptions="weighted centring; reflection corrected",
            debug_artifact="numeric/phase2_finite_rotation/finite_rotation_by_frame.csv",
            falsified_by="a known rotation not recovered, or det(R) = -1 accepted"),
        Requirement(
            key="5.7_one_dof_screw",
            requirement="One-DoF motion is a single screw",
            equation="xi_t = qdot_t xi_J, so X = [xi_1 ... xi_T] is rank 1",
            stage="common screw",
            module="joint_reduction.common_screw",
            symbol="twist_rank_evidence",
            fidelity="exact",
            uncertainty_source="per-frame twist information",
            assumptions="metric scales the translation half by the part radius",
            debug_artifact="numeric/common_screw/common_screw_fit.json",
            falsified_by="a two-DoF motion passing the rank-1 test"),
        Requirement(
            key="5.8_diff_finite_consistency",
            requirement="Differential and finite angular velocity must agree",
            equation="omega^diff ~ log(R_{t+1} R_t^T)/dt",
            stage="Phase 2 -> consistency",
            module="joint_reduction.differential_finite_consistency",
            symbol="compare_angular_velocity",
            fidelity="exact",
            uncertainty_source="RESIDUAL_CALIBRATED both sides; Mahalanobis when both exist",
            assumptions="spatial convention on both sides; midpoint stamping",
            debug_artifact="numeric/phase2_finite_rotation/angular_three_way.csv",
            falsified_by=("a known revolute trajectory whose two estimates "
                          "differ by more than the discretization order")),
        Requirement(
            key="5.9_joint_type",
            requirement="Joint type from rigid -> one-DoF -> P/R, not from a gain",
            equation="A: rigid? B: xi_t = qdot_t xi_J? C: omega = 0?",
            stage="geometric candidate",
            module="joint_reduction.direct_gaussian_joint",
            symbol="geometric_joint_candidate",
            fidelity="exact",
            uncertainty_source="chi-square quantiles of each statistic's own null",
            assumptions="one confidence and one mismatch_confidence for all scenes",
            debug_artifact="numeric/phase3_geometric_joint/joint_type_evidence.json",
            falsified_by=("a rigid one-DoF synthetic case reaching "
                          "MODEL_MISMATCH, or a two-body case reaching a type")),
        Requirement(
            key="5.10_renderer_null_test",
            requirement="Renderer confirms via an efficient score / null test on beta = 0",
            equation="H_0: beta = 0 at the best P candidate, nuisance-eliminated",
            stage="Branch C",
            module="joint_reduction.direct_rgbd_joint",
            symbol="rgbd_null_test_for_prismatic",
            fidelity="different",
            uncertainty_source="held-out camera folds; in-sample gain vs chi2(2)",
            assumptions=("compares refitted P and R losses rather than "
                         "evaluating the efficient score at the P optimum"),
            debug_artifact="numeric/branch_c_rgbd/",
            falsified_by=("showing the decision changes when the score is "
                          "computed exactly rather than by refit comparison")),
        Requirement(
            key="2_finite_vs_differential",
            requirement="The strain decomposition is applied to VELOCITY only",
            equation="sym(R - I) != 0 for finite rigid rotation; D = 0 for its velocity",
            stage="Phase 1C",
            module="joint_reduction.velocity_gradient_rigidity",
            symbol="fit_affine_velocity_field",
            fidelity="exact",
            uncertainty_source="n/a (a structural guarantee)",
            assumptions="the affine fit consumes velocities, never displacements",
            debug_artifact="numeric/phase1_differential/",
            falsified_by="feeding displacements and still reporting D = 0"),
        Requirement(
            key="9_velocity_covariance",
            requirement="Velocity uncertainty is propagated, not assumed isotropic",
            equation="Sigma_v = W Sigma_x W^T",
            stage="Phase 1B/1C",
            module="joint_reduction.velocity_gradient_rigidity",
            symbol="instantaneous_velocity",
            fidelity="approximate",
            uncertainty_source=("ANALYTIC_PROPAGATION when Phase 1 supplies "
                                "Sigma_x; IDENTITY_FALLBACK otherwise"),
            assumptions=("frames treated as independently estimated, so the "
                         "temporal cross-covariance W Sigma W^T off-diagonal "
                         "is dropped"),
            debug_artifact="numeric/phase1_free_motion/temporal_velocity_covariance_summary.json",
            falsified_by=("showing the dropped cross terms change a decision; "
                          "section 37 requires them to be reported")),
        Requirement(
            key="27_twist_convention",
            requirement="One twist convention, used consistently",
            equation="spatial: T_{t+1} = exp(dt xi^) T_t",
            stage="all",
            module="joint_reduction.rigid_twist_from_tracks",
            symbol="propagate_spatial_twist",
            fidelity="exact",
            uncertainty_source="n/a",
            assumptions="production uses the spatial convention",
            debug_artifact="numeric/phase2_rigid/",
            falsified_by="left and right composition giving the same result"),
        Requirement(
            key="29_gaussian_transport",
            requirement="Orientation and covariance move with the centre",
            equation="mu -> R mu + t, R^G -> R R^G, Sigma -> R Sigma R^T",
            stage="Phase 2 / Phase 4",
            module="joint_reduction.rigid_twist_from_tracks",
            symbol="propagate_gaussians",
            fidelity="exact",
            uncertainty_source="n/a",
            assumptions="spatial convention only",
            debug_artifact="renders/",
            falsified_by="an anisotropic splat keeping its orientation under rotation"),
        Requirement(
            key="35A_rigid_nested_test",
            requirement="Rigidity tested against a known null distribution",
            equation="(RSS_rigid - RSS_affine)/sigma^2 ~ chi2(6), sigma^2 from affine",
            stage="stage A",
            module="joint_reduction.velocity_gradient_rigidity",
            symbol="rigidity_likelihood_ratio",
            fidelity="exact",
            uncertainty_source="RESIDUAL_CALIBRATED from the affine (larger) model",
            assumptions=("MODEL_MISMATCH uses a far tighter confidence than "
                         "the P/R decision, because a chi2's power grows with N"),
            debug_artifact="numeric/phase1_differential/nested_test_by_frame.csv",
            falsified_by="a rigid synthetic support rejected by the test"),
        Requirement(
            key="8_floor_gain",
            requirement="The superseded gain-versus-floor comparison",
            equation="E_P - E_R vs epsilon_geom + epsilon_opt",
            stage="Phase 3 transform-only (legacy)",
            module="joint_reduction.geometric_joint_fit",
            symbol="decide_joint_type_continuous",
            fidelity="different",
            uncertainty_source="camera-fold spread of the gain, or (sigma/L)^2 in-run",
            assumptions=("section 38 says this is superseded by the physical "
                         "chain and retained as a diagnostic"),
            debug_artifact="numeric/phase3_geometric_joint/",
            falsified_by=("showing the two decision rules disagree on a scene "
                          "where the physical chain is well conditioned")),
    ]


def audit_row(req: Requirement) -> dict:
    exists, detail = _symbol_exists(req.module, req.symbol)
    reach = _reachable(req.symbol)
    trans = _transitively_production(req.symbol, req.module)
    # "Wired" means it runs in a standard batch run, not merely that an
    # argparse default would permit it.
    wired = bool(reach["production"]) or trans["runs_in_standard_batch"]
    tested = bool(reach["tests"])
    offline = bool(reach["tools"]) or (req.symbol in _tool_reachable_symbols())

    if not exists:
        status = MISSING
    elif req.fidelity == "different":
        status = DIFFERENT_STATISTIC
    elif not wired and not offline:
        status = NOT_WIRED
    elif not wired:
        status = DEBUG_ONLY
    elif req.fidelity == "approximate":
        status = APPROXIMATE
    elif tested:
        # Section 4: a passing synthetic test is not sufficient on its own.
        # It is sufficient only together with production reachability, which
        # is what `wired` has already established at this point.
        status = MATCHES_THEORY
    else:
        status = UNVERIFIED

    return {**asdict(req),
            "implemented": exists, "symbol_detail": detail,
            "referenced_in_production_entry": reach["production"],
            "referenced_in_tools": reach["tools"],
            "referenced_in_tests": reach["tests"],
            "production_wired": wired,
            "importable_from_production": trans["importable"],
            "on_argparse_default_path": trans["on_default_path"],
            "runs_in_standard_batch": trans["runs_in_standard_batch"],
            "called_by": trans["called_by"],
            "gated_by_flags": trans["gated_by"],
            "offline_available": offline,
            "has_test": tested,
            "status": status}


def build_matrix() -> list:
    return [audit_row(r) for r in requirements()]


def matrix_markdown(rows: list) -> str:
    out = ["# Theory / implementation matrix (section 4)", "",
           "Built by inspecting the source tree, not by assertion. "
           "`production_wired` means the symbol is reachable from "
           "`tools/kpart_pipeline.py` or `tools/run_iekf_scene.py`, directly "
           "or through the import graph. Per section 4, a passing synthetic "
           "test alone never earns `MATCHES_THEORY`.", "",
           "| requirement | equation | stage | module.symbol | fidelity | "
           "wired | test | status |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['requirement']} | `{r['equation']}` | {r['stage']} | "
            f"`{r['module']}.{r['symbol']}` | {r['fidelity']} | "
            f"{'yes' if r['production_wired'] else '**no**'} | "
            f"{'yes' if r['has_test'] else 'no'} | **{r['status']}** |")
    out += ["", "## Falsification criteria", "",
            "A requirement nobody could disprove is not being audited.", ""]
    for r in rows:
        out.append(f"- **{r['key']}** — {r['falsified_by']}")
    out += ["", "## Uncertainty source and assumptions", ""]
    for r in rows:
        out.append(f"- **{r['key']}** — uncertainty: {r['uncertainty_source']}; "
                   f"assumes: {r['assumptions']}")
    return "\n".join(out) + "\n"


def _defining_module(symbol: str) -> str:
    for folder in ("joint_reduction", "estimator", "debug", "tools"):
        for fp in (REPO / folder).glob("*.py"):
            try:
                src = fp.read_text()
            except Exception:
                continue
            if f"def {symbol}(" in src or f"class {symbol}" in src:
                return _repo_module_of(fp)
    return ""


def production_wiring_audit() -> dict:
    """Section 54: what the production path actually calls today."""
    checks = {
        "dual_differential_finite_evidence": "the section 33 physical chain",
        "geometric_joint_candidate": "the section 35 hierarchy",
        "rgbd_null_test_for_prismatic": "the section 34 renderer confirmation",
        "fit_joints": "the legacy Phase-3 transform-only joint fit",
        "decide_joint_type_continuous": "the continuous chart P/R decision",
        "discover_rigid_parts": "Phase-2 K-part discovery",
    }
    out = {}
    for sym, what in checks.items():
        reach = _reachable(sym)
        trans = _transitively_production(sym, _defining_module(sym))
        out[sym] = {"what": what,
                    "in_production_entry": reach["production"],
                    "in_tools": reach["tools"],
                    "in_tests": reach["tests"],
                    "importable_from_production": trans["importable"],
                    "called_by": trans["called_by"],
                    "gated_by_flags": trans["gated_by"],
                    "runs_by_default": trans["on_default_path"],
                    "runs_in_standard_batch": trans["runs_in_standard_batch"]}
    return out
