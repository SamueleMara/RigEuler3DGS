"""Section 40: the required plots.

CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md section 40.

Every plot is drawn from the same recorded state the numerical directory
writes, so a figure and its CSV cannot disagree. A plot whose data a run did
not produce is written as an explicit "not available at this stage" placeholder
rather than skipped -- section 13 requires the debug package to be complete
even when the run is unresolved, and a silently missing file reads as an
oversight instead of a stage that did not run.
"""

from __future__ import annotations

import math

import torch

REQUIRED = ["q_trajectories", "energy_by_stage", "support_counts",
            "support_jaccard", "rigid_residual_by_frame",
            "pairwise_invariant_error", "inertia_eigenvalue_drift",
            "prismatic_residuals", "revolute_residuals", "axis_stability",
            "multiview_q_proposals", "fisher_condition",
            "renderer_refinement"]


def write_plots(dbg, writer, energies=None) -> list:
    """Write section 40's figures into `plots/`. Returns the filenames."""
    if not dbg.enabled:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = writer.dir / "plots"
    root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "figure.autolayout": True})
    written = []

    def fig(name, draw, title, ylabel=""):
        f, ax = plt.subplots(figsize=(6.2, 3.1))
        ok = False
        try:
            ok = draw(ax)
        except Exception as exc:                      # a bad plot must not
            ax.text(0.5, 0.5, f"plot failed: {exc}",  # take the run down
                    ha="center", va="center", transform=ax.transAxes,
                    color="crimson", fontsize=7)
        if ok is False:
            ax.text(0.5, 0.5, "not available at this stage", ha="center",
                    va="center", transform=ax.transAxes, color="#888")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel)
        p = root / f"{name}.png"
        f.savefig(p)
        plt.close(f)
        written.append(f"plots/{p.name}")

    T = dbg.num_frames
    xs = list(range(T))

    # ---- q trajectories -------------------------------------------------
    def _q(ax):
        any_ = False
        for k in range(1, dbg.K + 1):
            j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
            if j and j.get("q"):
                ax.plot(range(len(j["q"])), j["q"], "--", lw=1,
                        label=f"part {k} geometric")
                any_ = True
            if k - 1 < len(dbg.q_traj) and dbg.q_traj[k - 1]:
                ax.plot(range(len(dbg.q_traj[k - 1])), dbg.q_traj[k - 1],
                        lw=1.6, label=f"part {k} renderer-refined")
                any_ = True
        if any_:
            ax.set_xlabel("frame")
            ax.legend(fontsize=7)
        return any_
    fig("q_trajectories", _q, "q(t) by stage (§29)", "q")

    # ---- energy by stage -------------------------------------------------
    def _e(ax):
        if not energies:
            return False
        any_ = False
        for name in ("canonical", "bootstrap", "rigid", "refined"):
            v = [(e or {}).get("total") for e in energies.get(name, [])]
            v = [x for x in v if x is not None]
            if v:
                ax.plot(range(len(v)), v, lw=1.4, label=name)
                any_ = True
        if any_:
            ax.set_xlabel("frame")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        return any_
    fig("energy_by_stage", _e, "RGB-D energy per frame by stage (§35/§36)", "E")

    # ---- support counts / jaccard ---------------------------------------
    def _c(ax):
        if not dbg.rigid_history:
            return False
        it = [h.get("outer") for h in dbg.rigid_history if "counts" in h]
        cnt = [h["counts"] for h in dbg.rigid_history if "counts" in h]
        if not cnt:
            return False
        for k in range(len(cnt[0])):
            ax.plot(it, [c[k] for c in cnt], marker="o", ms=3,
                    label="static" if k == 0 else f"part {k}")
        ax.set_xlabel("outer iteration")
        ax.legend(fontsize=7)
        return True
    fig("support_counts", _c, "Label counts per outer iteration (§20)", "count")

    def _j(ax):
        v = [(h.get("outer"), h.get("label_jaccard"))
             for h in dbg.rigid_history if h.get("label_jaccard") is not None]
        if not v:
            return False
        ax.plot([a for a, _ in v], [b for _, b in v], marker="o", ms=3)
        ax.axhline(0.98, ls=":", c="crimson", lw=1, label="§28 tau_z")
        ax.set_xlabel("outer iteration")
        ax.legend(fontsize=7)
        return True
    fig("support_jaccard", _j, "Support Jaccard per outer iteration (§20)",
        "Jaccard")

    # ---- rigid residual by frame ----------------------------------------
    def _r(ax):
        if not dbg.positions or dbg.transforms is None or dbg.labels is None:
            return False
        med, p95 = [], []
        for t in range(min(T, len(dbg.positions))):
            pred = dbg.positions_at("rigid", t)
            if pred is None:
                break
            r = (dbg.positions[t].double() - pred.double()).norm(dim=1)
            r = r[dbg.labels > 0]
            if r.numel() == 0:
                break
            med.append(float(r.median()))
            p95.append(float(r.quantile(0.95)))
        if not med:
            return False
        ax.plot(range(len(med)), med, lw=1.5, label="median")
        ax.plot(range(len(p95)), p95, lw=1.0, ls="--", label="p95")
        ax.set_xlabel("frame")
        ax.legend(fontsize=7)
        return True
    fig("rigid_residual_by_frame", _r, "Per-frame rigid residual (§22)",
        "world units")

    # ---- invariants ------------------------------------------------------
    def _inv(ax, which):
        if dbg.labels is None or not dbg.positions or dbg.mu0 is None:
            return False
        g = torch.Generator(device="cpu").manual_seed(0)
        any_ = False
        for k in range(1, dbg.K + 1):
            m = dbg.labels == k
            n = int(m.sum())
            if n < 3:
                continue
            x0 = dbg.mu0[m].double()
            np_ = min(2048, n * (n - 1) // 2)
            i = torch.randint(0, n, (np_,), generator=g).to(x0.device)
            jj = torch.randint(0, n, (np_,), generator=g).to(x0.device)
            keep = i != jj
            i, jj = i[keep], jj[keep]
            d0 = (x0[i] - x0[jj]).norm(dim=1)
            c0 = x0.mean(0)
            e0 = torch.linalg.eigvalsh((x0 - c0).T @ (x0 - c0))
            ys = []
            for t in range(min(T, len(dbg.positions))):
                xt = dbg.positions[t][m].double()
                if which == "pair":
                    ys.append(float((xt[i] - xt[jj]).norm(dim=1).sub(d0)
                                    .abs().mean()))
                else:
                    ct = xt.mean(0)
                    et = torch.linalg.eigvalsh((xt - ct).T @ (xt - ct))
                    ys.append(float(((et - e0).abs()
                                     / e0.clamp_min(1e-12)).max()))
            if ys:
                ax.plot(range(len(ys)), ys, lw=1.4, label=f"part {k}")
                any_ = True
        if any_:
            ax.set_xlabel("frame")
            ax.legend(fontsize=7)
        return any_
    fig("pairwise_invariant_error", lambda a: _inv(a, "pair"),
        "Pairwise distance drift (§23)", "mean |dd|")
    fig("inertia_eigenvalue_drift", lambda a: _inv(a, "inertia"),
        "Second-moment eigenvalue drift (§24)", "max relative drift")

    # ---- prismatic / revolute per-frame residuals ------------------------
    def _fit(ax, kind):
        any_ = False
        for k in range(1, dbg.K + 1):
            j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
            if not j:
                continue
            f = j.get(kind) or {}
            pf = f.get("per_frame_error")
            if pf:
                ax.plot(range(len(pf)), pf, lw=1.4,
                        label=f"part {k}" + (" (selected)"
                                             if j.get("selected") == kind else ""))
                any_ = True
        if any_:
            ax.set_xlabel("frame")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        return any_
    fig("prismatic_residuals", lambda a: _fit(a, "prismatic"),
        "Prismatic fit residual per frame (§25)", "E^geom")
    fig("revolute_residuals", lambda a: _fit(a, "revolute"),
        "Revolute fit residual per frame (§26)", "E^geom")

    # ---- axis stability --------------------------------------------------
    def _ax(ax):
        any_ = False
        for k in range(1, dbg.K + 1):
            j = dbg.joints[k - 1] if k - 1 < len(dbg.joints) else None
            it = ((j or {}).get("revolute") or {}).get("axis_iterates") or []
            if len(it) < 2:
                continue
            d = []
            for i in range(1, len(it)):
                a = torch.tensor(it[i - 1]["axis"], dtype=torch.float64)
                b = torch.tensor(it[i]["axis"], dtype=torch.float64)
                d.append(math.acos(float((a @ b).abs().clamp(max=1.0))))
            ax.semilogy(range(1, len(d) + 1), [max(x, 1e-18) for x in d],
                        marker="o", ms=3, label=f"part {k}")
            any_ = True
        if any_:
            ax.set_xlabel("revolute-fit iteration")
            ax.legend(fontsize=7)
        return any_
    fig("axis_stability", _ax, "Axis direction change per iteration (§27)",
        "rad")

    # ---- multiview ------------------------------------------------------
    def _mv(ax):
        if not dbg.multiview:
            return False
        cams = sorted({r["camera_id"] for r in dbg.multiview})
        for c in cams:
            rs = [r for r in dbg.multiview if r["camera_id"] == c]
            ax.plot([r["frame"] for r in rs], [r.get("delta_q", 0) for r in rs],
                    marker=".", ms=3, lw=1, label=f"cam {c}")
        ax.axhline(0, c="#888", lw=0.8)
        ax.set_xlabel("frame")
        ax.legend(fontsize=6, ncol=2)
        return True
    fig("multiview_q_proposals", _mv,
        "Per-camera dq = g/(h+eps) (§32)", "dq")

    # ---- fisher condition ------------------------------------------------
    def _fc(ax):
        rs = [r for r in dbg.camera_rows if r.get("fisher_trace") is not None]
        if not rs:
            return False
        cams = sorted({r["camera_id"] for r in rs})
        for c in cams:
            v = [r for r in rs if r["camera_id"] == c]
            ax.semilogy([r["frame"] for r in v],
                        [max(r["fisher_trace"], 1e-30) for r in v],
                        marker=".", ms=3, lw=1, label=f"cam {c}")
        ax.set_xlabel("frame")
        ax.legend(fontsize=6, ncol=2)
        return True
    fig("fisher_condition", _fc, "Per-camera Fisher trace (§16/§33)", "trace")

    # ---- renderer refinement --------------------------------------------
    def _rr(ax):
        it = (dbg.refine_info or {}).get("trajectory_iterations") or []
        if not it:
            return False
        ax.plot([i.get("iter", n) for n, i in enumerate(it)],
                [i.get("energy") for i in it], marker="o", ms=3)
        ax.set_xlabel("Levenberg iteration")
        return True
    fig("renderer_refinement", _rr,
        "Renderer refinement objective (§30)", "E")

    writer.json("plots_index", {"note": "section 40", "files": written})
    return written
