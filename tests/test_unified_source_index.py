"""T1 — two-source candidate provenance.

CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md section 2, test T1.

Unequal counts, shuffled rows, no assumed correspondence. The population is a
candidate state space; the tests check that identity survives it and that the
manifest refuses to describe a raw union as a solved scene.

    python tests/test_unified_source_index.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bidirectional_iekf.provenance import (  # noqa: E402
    FINAL,
    INITIAL,
    SourceAnchor,
    build_candidate_population,
    candidate_manifest,
    write_population,
)

CHECKS = 0
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")


def raises(label: str, fn, needle: str = "") -> None:
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        if needle and needle.lower() not in str(exc).lower():
            FAILURES.append(f"{label}: message lacks {needle!r}: {exc}")
            print(f"  FAIL  {label}  message lacks {needle!r}")
        return
    FAILURES.append(f"{label}: no exception raised")
    print(f"  FAIL  {label}  no exception raised")


def anchor(endpoint, n, frame, seed=0):
    rng = np.random.default_rng(seed)
    return SourceAnchor(endpoint=endpoint, model_path=f"/models/{endpoint}",
                        ply_path=f"/models/{endpoint}/pc.ply",
                        ply_sha256="0" * 64, anchor_original_frame=frame,
                        xyz=rng.normal(size=(n, 3)), num_gaussians=n)


def test_unequal_counts() -> None:
    print("[1] unequal source sizes, washer's real numbers")
    a0, aT = anchor(INITIAL, 21338, 0, 1), anchor(FINAL, 24582, 59, 2)
    pop = build_candidate_population([a0, aT])
    check("candidates are the disjoint union", pop.num_candidates == 45920,
          str(pop.num_candidates))
    check("initial block size", int(pop.mask_for(INITIAL).sum()) == 21338)
    check("final block size", int(pop.mask_for(FINAL).sum()) == 24582)
    check("initial anchors at frame 0",
          set(pop.anchor_original_frame[pop.mask_for(INITIAL)].tolist()) == {0})
    check("final anchors at frame 59",
          set(pop.anchor_original_frame[pop.mask_for(FINAL)].tolist()) == {59})

    rows_init = pop.source_gaussian_index[pop.mask_for(INITIAL)]
    check("source rows are 0..N0-1 exactly",
          np.array_equal(np.sort(rows_init), np.arange(21338)))
    rows_fin = pop.source_gaussian_index[pop.mask_for(FINAL)]
    check("final rows restart at 0",
          rows_fin.min() == 0 and rows_fin.max() == 24581)

    xyz_back = pop.anchor_xyz[pop.rows_for(INITIAL)]
    check("initial geometry survives losslessly",
          np.array_equal(xyz_back, a0.xyz), "anchors must not be modified")
    check("final geometry survives losslessly",
          np.array_equal(pop.anchor_xyz[pop.rows_for(FINAL)], aT.xyz))


def test_shuffled_rows_keep_identity() -> None:
    print("[2] shuffled source rows keep their identity")
    rng = np.random.default_rng(7)
    base = anchor(INITIAL, 50, 0, 3)
    order = rng.permutation(50)
    shuffled = SourceAnchor(endpoint=INITIAL, model_path=base.model_path,
                            ply_path=base.ply_path, ply_sha256=base.ply_sha256,
                            anchor_original_frame=0, xyz=base.xyz[order],
                            num_gaussians=50)
    pop = build_candidate_population([shuffled, anchor(FINAL, 20, 59, 4)])
    # Row k of the population is row k of the (already shuffled) source: the
    # mapping is positional and recorded, never re-derived by matching.
    check("row k maps to source row k",
          np.array_equal(pop.anchor_xyz[pop.rows_for(INITIAL)], shuffled.xyz))
    check("a shuffle changes the geometry but not the index set",
          not np.array_equal(shuffled.xyz, base.xyz)
          and np.array_equal(np.sort(pop.source_gaussian_index[pop.mask_for(INITIAL)]),
                             np.arange(50)))


def test_hash_and_manifest() -> None:
    print("[3] index hash and the manifest's refusals")
    pop_a = build_candidate_population([anchor(INITIAL, 30, 0, 5),
                                        anchor(FINAL, 40, 59, 6)])
    pop_b = build_candidate_population([anchor(INITIAL, 30, 0, 5),
                                        anchor(FINAL, 40, 59, 6)])
    pop_c = build_candidate_population([anchor(INITIAL, 31, 0, 5),
                                        anchor(FINAL, 40, 59, 6)])
    check("identical populations hash identically",
          pop_a.index_hash() == pop_b.index_hash())
    check("a different size changes the hash",
          pop_a.index_hash() != pop_c.index_hash())

    manifest = candidate_manifest(pop_a, total_frames=60,
                                  source_pose_treatment="conditional_fixed_anchor")
    check("manifest records no assumed correspondence",
          manifest["correspondence_assumed"] is False)
    check("manifest defaults the representation gate to NOT_RUN",
          manifest["representation_gate"] == "NOT_RUN")
    check("manifest states the union is not a solved scene",
          "not a demonstrated single scene" in manifest["note"])
    check("manifest carries both source hashes",
          set(manifest["sources"]) == {INITIAL, FINAL})

    # Section 2: the source-pose model must be declared, never implied.
    raises("an undeclared source-pose model is refused",
           lambda: candidate_manifest(pop_a, total_frames=60,
                                      source_pose_treatment="whatever"),
           "must say which model is in force")

    raises("a duplicated endpoint is refused",
           lambda: build_candidate_population(
               [anchor(INITIAL, 5, 0), anchor(INITIAL, 6, 0)]),
           "once")

    with tempfile.TemporaryDirectory() as tmp:
        written = write_population(Path(tmp), pop_a, manifest)
        saved = json.loads((Path(tmp) / "candidate_manifest.json").read_text())
        check("manifest written atomically", saved == written)
        check("arrays written",
              np.load(Path(tmp) / "source_endpoint.npy").shape[0] == 70
              and np.load(Path(tmp) / "source_gaussian_index.npy").shape[0] == 70)
        check("no temp file left behind",
              not (Path(tmp) / "candidate_manifest.json.tmp").exists())


def main() -> int:
    test_unequal_counts()
    test_shuffled_rows_keep_identity()
    test_hash_and_manifest()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
