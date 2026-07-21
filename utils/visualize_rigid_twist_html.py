"""Plot explicit learned omega(t), u(t) trajectories from gaussian_tracks.pt."""

import json
from pathlib import Path
import torch


def write_rigid_twist_html_from_file(tracks_path, out_html, title="Learned rigid twist"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tracks = torch.load(str(tracks_path), map_location="cpu")
    times = tracks.get("times", None)
    omega = tracks.get("rigid_omega", None)
    trans = tracks.get("rigid_translation", None)
    if not (torch.is_tensor(times) and torch.is_tensor(omega) and torch.is_tensor(trans)):
        return None
    t = times.detach().float().reshape(-1).numpy()
    w = omega.detach().float().reshape(-1, 3).numpy()
    u = trans.detach().float().reshape(-1, 3).numpy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Angular velocity omega(t)", "Translation velocity u(t)"))
    axes = ("x", "y", "z")
    for j, name in enumerate(axes):
        fig.add_trace(go.Scatter(x=t, y=w[:, j], mode="lines+markers", name=f"omega_{name}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=u[:, j], mode="lines+markers", name=f"u_{name}"), row=2, col=1)
    fig.update_xaxes(title_text="normalized time", row=2, col=1)
    fig.update_yaxes(title_text="angular velocity", row=1, col=1)
    fig.update_yaxes(title_text="translation velocity", row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", height=760, hovermode="x unified")
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    summary = {
        "html": str(out),
        "num_times": int(len(t)),
        "omega_norm_min": float(torch.linalg.norm(omega.float(), dim=-1).min().item()),
        "omega_norm_max": float(torch.linalg.norm(omega.float(), dim=-1).max().item()),
        "translation_norm_min": float(torch.linalg.norm(trans.float(), dim=-1).min().item()),
        "translation_norm_max": float(torch.linalg.norm(trans.float(), dim=-1).max().item()),
    }
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return out
