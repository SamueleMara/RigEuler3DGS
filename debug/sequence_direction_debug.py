"""Debug output for the sequence-direction phase.

CODEX_BIDIRECTIONAL_IEKF_AND_ARTICULATED_3DGS_MERGE.md sections 4-5.

What can go wrong here is silent: a run that reverses the frames but keeps
forward numbering, or that reverses the ids and not the pixels, produces a
trajectory that looks plausible and is indexed wrongly. Every later
forward/backward comparison then measures the bug. So this phase writes a
table that can be read directly against the spec -- which frame was processed
when, what the dataset calls it, and what dt was used between steps -- plus a
page that draws the mapping so an off-by-one is visible rather than arithmetic.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

FIELDS = ("processing_index", "original_frame_index", "dt_used_s",
          "rgb_frame_id", "depth_frame_id", "depth_valid_pixels")


def _rows(frame_ids):
    rows = []
    for entry in frame_ids.get("frames", []):
        cameras = entry.get("cameras") or []
        first = cameras[0] if cameras else {}
        rows.append({
            "processing_index": entry.get("processing_index", entry.get("frame")),
            "original_frame_index": entry.get(
                "original_frame_index", entry.get("dataset_frame_index")),
            "dt_used_s": entry.get("dt_used_s"),
            "rgb_frame_id": first.get("rgb_frame_id"),
            "depth_frame_id": first.get("depth_frame_id"),
            "depth_valid_pixels": sum(int(c.get("depth_valid_pixels") or 0)
                                      for c in cameras),
        })
    return rows


def write_frame_order_csv(path, frame_ids) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in _rows(frame_ids):
            writer.writerow(row)
    return path


def check_frame_order(frame_ids) -> dict:
    """Assertions a reader would otherwise have to make by eye.

    Returned rather than raised: this is a debug artifact, and a run that
    fails one of these should still write its evidence so the failure can be
    read. The caller decides whether to stop.
    """
    rows = _rows(frame_ids)
    direction = frame_ids.get("sequence_direction", "forward")
    originals = [r["original_frame_index"] for r in rows]
    dts = [r["dt_used_s"] for r in rows if r["dt_used_s"] is not None]
    expected = (sorted(originals) if direction == "forward"
                else sorted(originals, reverse=True))
    checks = {
        "direction": direction,
        "num_frames": len(rows),
        "processing_index_is_contiguous":
            [r["processing_index"] for r in rows] == list(range(len(rows))),
        "original_indices_unique": len(set(originals)) == len(originals),
        "original_order_matches_direction": originals == expected,
        "all_dt_positive": all(float(d) > 0 for d in dts),
        "first_original_frame": originals[0] if originals else None,
        "last_original_frame": originals[-1] if originals else None,
        "every_frame_has_rgb": all(r["rgb_frame_id"] for r in rows),
        "frames_with_depth": sum(1 for r in rows if r["depth_frame_id"]),
    }
    checks["passes"] = all(
        checks[k] for k in ("processing_index_is_contiguous",
                            "original_indices_unique",
                            "original_order_matches_direction",
                            "all_dt_positive", "every_frame_has_rgb"))
    return checks


def write_frame_order_html(path, frame_ids, scene: str = "") -> Path:
    """A page that draws the mapping, because an off-by-one is visual."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows(frame_ids)
    checks = check_frame_order(frame_ids)
    direction = checks["direction"]
    total = max(len(rows), 1)

    lines = []
    for row in rows:
        processing = int(row["processing_index"])
        original = int(row["original_frame_index"])
        x1 = 40 + processing * (720 / total)
        x2 = 40 + original * (720 / total)
        lines.append(
            f"<line x1='{x1:.1f}' y1='30' x2='{x2:.1f}' y2='150' "
            f"stroke='var(--accent)' stroke-opacity='0.45' stroke-width='1.2'/>")
    svg = (
        "<svg viewBox='0 0 800 190' role='img' "
        f"aria-label='processing order mapped to original frame index, {direction}'>"
        "<text x='40' y='18' font-size='12' fill='var(--muted)'>processing order"
        "</text>" + "".join(lines) +
        "<text x='40' y='176' font-size='12' fill='var(--muted)'>"
        "original frame index</text></svg>")

    def cell(value):
        return html.escape("" if value is None else str(value))

    body = "".join(
        "<tr>" + "".join(f"<td>{cell(row[f])}</td>" for f in FIELDS) + "</tr>"
        for row in rows)
    head = "".join(f"<th>{html.escape(f)}</th>" for f in FIELDS)
    verdict = "PASS" if checks["passes"] else "FAIL"
    checklist = "".join(
        f"<li><code>{html.escape(k)}</code>: <b>{html.escape(str(v))}</b></li>"
        for k, v in checks.items() if k != "direction")

    path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frame order - {html.escape(scene or direction)}</title>
<style>
  :root {{ --bg:#fff; --fg:#111; --muted:#666; --line:#e3e3e3; --accent:#2563eb;
           --ok:#15803d; --bad:#b91c1c; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --bg:#0e0e10; --fg:#ececec; --muted:#9a9a9a; --line:#2a2a2e; --accent:#60a5fa;
    --ok:#4ade80; --bad:#f87171; }} }}
  :root[data-theme="dark"] {{ --bg:#0e0e10; --fg:#ececec; --muted:#9a9a9a;
    --line:#2a2a2e; --accent:#60a5fa; --ok:#4ade80; --bad:#f87171; }}
  body {{ background:var(--bg); color:var(--fg); margin:0; padding:24px 16px;
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width:860px; margin:0 auto; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  p.sub {{ color:var(--muted); margin:0 0 20px; }}
  .verdict {{ font-weight:700; color:{'var(--ok)' if checks['passes'] else 'var(--bad)'}; }}
  svg {{ width:100%; height:auto; border:1px solid var(--line); border-radius:8px;
    background:var(--bg); margin:12px 0 20px; }}
  ul {{ margin:0 0 20px; padding-left:20px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ border-bottom:1px solid var(--line); padding:5px 8px; text-align:left;
    white-space:nowrap; }}
  th {{ color:var(--muted); font-weight:600; }}
  .wrap {{ overflow-x:auto; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
</style></head><body><main>
<h1>Frame order &mdash; {html.escape(direction)}</h1>
<p class="sub">{html.escape(scene)} &middot; {len(rows)} frames &middot;
checks <span class="verdict">{verdict}</span></p>
{svg}
<ul>{checklist}</ul>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
</main></body></html>""", encoding="utf-8")
    return path


def write_phase_debug(out_dir, frame_ids, scene: str = "") -> dict:
    """Everything this phase owes: csv, page, and a machine-readable verdict."""
    out_dir = Path(out_dir)
    checks = check_frame_order(frame_ids)
    csv_path = write_frame_order_csv(out_dir / "frame_order.csv", frame_ids)
    html_path = write_frame_order_html(out_dir / "frame_order.html", frame_ids, scene)
    json_path = out_dir / "frame_order_checks.json"
    json_path.write_text(json.dumps(
        {**checks,
         "canonical_endpoint": frame_ids.get("canonical_endpoint"),
         "processing_order_original_indices":
             frame_ids.get("processing_order_original_indices"),
         "csv": str(csv_path), "html": str(html_path)}, indent=1))
    return {"checks": checks, "csv": csv_path, "html": html_path, "json": json_path}
