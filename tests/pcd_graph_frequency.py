#!/usr/bin/env python3
"""
pcd_graph_frequency.py

Independent spatial graph-frequency analysis for PLY point clouds.

NO point correspondence is required between input PLYs. Each cloud is
decomposed independently.

For each cloud:
  1. Load XYZ.
  2. Build a Gaussian-weighted spatial k-NN graph.
  3. Compute the normalized graph Laplacian:
         L = I - D^{-1/2} W D^{-1/2}
  4. Compute the lowest M graph frequencies:
         L u_k = lambda_k u_k
  5. Treat centered XYZ as graph signals and compute:
         X_hat_k = u_k^T X
  6. Compute LOCAL spectral geometry energy:
         e_k(i) = u_k(i)^2 * ||X_hat_k||^2
  7. The HTML lets you choose a frequency BAND [k_low, k_high].
     Each point is scored by:
         s_i = sum_{k=k_low}^{k_high} e_k(i)
     The selected points are those with the highest band energy.

This is what "which points are taken by this frequency range?" means in the
viewer.

The script safely handles multiple inputs with the same basename, e.g.:

  .../frame0_static/point_cloud/iteration_14000/point_cloud.ply
  .../frame_t_static/point_cloud/iteration_14000/point_cloud.ply

Outputs are named using the meaningful parent directory:

  frame0_static_graph_frequency.html
  frame0_static_graph_spectrum.npz
  frame_t_static_graph_frequency.html
  frame_t_static_graph_spectrum.npz

Dependencies:
  pip install numpy scipy open3d plotly tqdm
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh
from tqdm.auto import tqdm


DEFAULT_INPUTS = [
    "/home/samuelemara/Joint3DGS/RigEuler3DGS/output/"
    "storage_45135_static_endpoints_14k/shared/frame0_static/"
    "point_cloud/iteration_14000/point_cloud.ply",

    "/home/samuelemara/Joint3DGS/RigEuler3DGS/output/"
    "storage_45135_static_endpoints_14k/shared/frame_t_static/"
    "point_cloud/iteration_14000/point_cloud.ply",
]

DEFAULT_OUTPUT = (
    "/home/samuelemara/Joint3DGS/RigEuler3DGS/output/"
    "storage_45135_static_endpoints_14k/graph_frequency"
)


# =============================================================================
# Naming
# =============================================================================

GENERIC_DIR_PATTERNS = [
    re.compile(r"^point_cloud$", re.I),
    re.compile(r"^points?$", re.I),
    re.compile(r"^iteration[_-]?\d+$", re.I),
    re.compile(r"^iter[_-]?\d+$", re.I),
    re.compile(r"^shared$", re.I),
]


def is_generic_dir(name: str) -> bool:
    return any(p.fullmatch(name) for p in GENERIC_DIR_PATTERNS)


def derive_cloud_label(path: Path) -> str:
    """
    Example:
      .../frame0_static/point_cloud/iteration_14000/point_cloud.ply
      -> frame0_static
    """
    p = path.expanduser().resolve()

    # If filename itself is meaningful, use it.
    if p.stem.lower() not in {"point_cloud", "cloud", "points"}:
        return p.stem

    # Strong preference for frame-like ancestors.
    for parent in p.parents:
        n = parent.name
        if re.match(r"^frame", n, flags=re.I):
            return n

    # Otherwise take nearest non-generic parent.
    for parent in p.parents:
        n = parent.name
        if n and not is_generic_dir(n):
            return n

    return p.stem


def make_unique_labels(paths):
    raw = [derive_cloud_label(Path(p)) for p in paths]

    result = []
    used = {}

    for base in raw:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")
        if not safe:
            safe = "cloud"

        count = used.get(safe, 0) + 1
        used[safe] = count

        result.append(safe if count == 1 else f"{safe}_{count}")

    # Explicit collision guard.
    if len(set(result)) != len(result):
        raise RuntimeError(f"Internal label collision: {result}")

    return result


# =============================================================================
# PLY
# =============================================================================

def load_ply_xyz(path: str):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError(f"Invalid or empty PLY: {path}")

    return pcd, xyz


def deterministic_subsample(xyz, max_points, seed):
    n = len(xyz)

    if max_points <= 0 or n <= max_points:
        idx = np.arange(n, dtype=np.int64)
        return xyz, idx

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return xyz[idx], idx


# =============================================================================
# Graph
# =============================================================================

def build_knn_graph(
    xyz,
    k,
    sigma=None,
    chunk_size=8192,
    desc="k-NN",
):
    n = len(xyz)

    if n < 3:
        raise ValueError("Need at least 3 points.")

    k = int(min(max(1, k), n - 1))
    chunk_size = max(1, int(chunk_size))

    tree = cKDTree(xyz)

    row_chunks = []
    col_chunks = []
    dist_chunks = []

    starts = list(range(0, n, chunk_size))

    for start in tqdm(
        starts,
        desc=desc,
        unit="chunk",
        dynamic_ncols=True,
        leave=False,
    ):
        end = min(start + chunk_size, n)

        d, nn = tree.query(
            xyz[start:end],
            k=k + 1,
            workers=-1,
        )

        # Remove self-neighbor.
        d = d[:, 1:]
        nn = nn[:, 1:]

        rows = np.repeat(
            np.arange(start, end, dtype=np.int64),
            k,
        )

        row_chunks.append(rows)
        col_chunks.append(nn.reshape(-1).astype(np.int64))
        dist_chunks.append(d.reshape(-1))

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    distances = np.concatenate(dist_chunks)

    if sigma is None:
        positive = distances[distances > 0]
        sigma = float(np.median(positive)) if len(positive) else 1.0

    sigma = max(float(sigma), 1e-12)

    weights = np.exp(
        -(distances ** 2) / (2.0 * sigma ** 2)
    )

    W = coo_matrix(
        (weights, (rows, cols)),
        shape=(n, n),
        dtype=np.float64,
    ).tocsr()

    # Symmetrize.
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()

    return W, sigma


def graph_spectrum(W, n_modes, normalized=True, tol=1e-5):
    n = W.shape[0]
    n_modes = min(max(2, int(n_modes)), n - 1)

    L = csgraph.laplacian(W, normed=normalized)

    vals, vecs = eigsh(
        L,
        k=n_modes,
        which="SM",
        tol=tol,
    )

    order = np.argsort(vals)

    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])

    vals[np.abs(vals) < 1e-12] = 0.0

    return vals, vecs


# =============================================================================
# Geometry spectrum
# =============================================================================

def compute_geometry_spectrum(xyz, eigvecs):
    """
    centered XYZ is a 3-channel graph signal.

      coeff[k,:] = u_k^T X

    Global geometry energy:
      E_k = ||coeff[k,:]||^2

    Local energy:
      e_k(i) = u_k(i)^2 E_k

    Because u_k is L2 normalized:
      sum_i e_k(i) = E_k

    For a selected band B:
      score_i(B) = sum_{k in B} e_k(i)

    This is the quantity shown in the HTML.
    """
    centered = xyz - xyz.mean(axis=0, keepdims=True)

    coeff = eigvecs.T @ centered               # M x 3
    global_energy = np.sum(coeff ** 2, axis=1) # M

    total_global = float(global_energy.sum())

    if total_global > 0:
        global_energy_norm = global_energy / total_global
    else:
        global_energy_norm = np.zeros_like(global_energy)

    cumulative_global = np.cumsum(global_energy_norm)

    # N x M
    local_energy = (eigvecs ** 2) * global_energy[None, :]

    total_local = float(local_energy.sum())
    if total_local > 0:
        local_energy_norm = local_energy / total_local
    else:
        local_energy_norm = np.zeros_like(local_energy)

    # Prefix in mode direction:
    # prefix[:, q] = energy from modes [0, q-1]
    # shape N x (M+1)
    prefix = np.concatenate(
        [
            np.zeros((len(xyz), 1), dtype=np.float64),
            np.cumsum(local_energy_norm, axis=1),
        ],
        axis=1,
    )

    return (
        coeff,
        global_energy,
        global_energy_norm,
        cumulative_global,
        local_energy_norm,
        prefix,
    )


# =============================================================================
# HTML
# =============================================================================

def write_html(
    path,
    label,
    source_path,
    xyz,
    eigvals,
    global_energy_norm,
    cumulative_global,
    local_energy_prefix,
    knn,
    sigma,
    original_count,
):
    from plotly.offline import get_plotlyjs

    n_modes = len(eigvals)

    # Browser only needs XYZ, frequencies, spectrum, and local-energy prefix.
    # Store prefix mode-major for convenient JS indexing:
    # prefixMode[q][i].
    prefix_mode_major = []

    for q in tqdm(
        range(n_modes + 1),
        desc=f"{label}: HTML band data",
        unit="freq",
        leave=False,
        dynamic_ncols=True,
    ):
        prefix_mode_major.append(
            local_energy_prefix[:, q].astype(np.float32).tolist()
        )

    payload = {
        "xyz": xyz.astype(np.float32).tolist(),
        "eigenvalues": eigvals.astype(np.float32).tolist(),
        "globalEnergy": global_energy_norm.astype(np.float32).tolist(),
        "cumulativeGlobal": cumulative_global.astype(np.float32).tolist(),
        "prefixLocalEnergy": prefix_mode_major,
        "source": str(source_path),
        "label": label,
        "knn": int(knn),
        "sigma": float(sigma),
        "originalCount": int(original_count),
        "usedCount": int(len(xyz)),
    }

    tqdm.write(f"  [{label}] serializing HTML...")
    payload_json = json.dumps(payload, separators=(",", ":"))

    plotly_js = get_plotlyjs()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Graph frequency — {label}</title>
<script>{plotly_js}</script>
<style>
:root {{
  color-scheme: dark;
  --bg:#101318;
  --panel:#181d23;
  --panel2:#20262e;
  --border:#303945;
  --text:#edf2f7;
  --muted:#9aa7b5;
  --accent:#f59e0b;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;
}}
main {{
  max-width:1500px;
  margin:auto;
  padding:16px;
}}
h1 {{
  margin:0 0 4px;
  font-size:21px;
}}
.source {{
  font-size:12px;
  color:var(--muted);
  word-break:break-all;
  margin-bottom:14px;
}}
.controls {{
  padding:14px;
  border:1px solid var(--border);
  border-radius:12px;
  background:var(--panel);
  margin-bottom:14px;
}}
.row {{
  display:grid;
  grid-template-columns:180px minmax(200px,1fr) 240px;
  align-items:center;
  gap:12px;
  margin:9px 0;
}}
@media(max-width:760px) {{
  .row {{ grid-template-columns:1fr; }}
}}
label {{
  color:var(--muted);
  font-size:13px;
}}
input[type=range] {{
  width:100%;
}}
button, select {{
  border:1px solid var(--border);
  border-radius:8px;
  color:var(--text);
  background:var(--panel2);
  padding:8px 10px;
  min-height:38px;
}}
button {{ cursor:pointer; }}
.value {{
  font-size:13px;
  font-variant-numeric:tabular-nums;
}}
.stats {{
  display:flex;
  flex-wrap:wrap;
  gap:18px;
  margin-top:12px;
  color:var(--muted);
  font-size:13px;
}}
.stats b {{ color:var(--text); }}
.grid {{
  display:grid;
  grid-template-columns:1.45fr 1fr;
  gap:14px;
}}
@media(max-width:950px) {{
  .grid {{ grid-template-columns:1fr; }}
}}
.panel {{
  border:1px solid var(--border);
  border-radius:12px;
  background:var(--panel);
  min-width:0;
  padding:8px;
}}
#cloud {{ height:720px; }}
#spectrum {{ height:420px; }}
.help {{
  margin-top:10px;
  color:var(--muted);
  font-size:12px;
  line-height:1.5;
}}
</style>
</head>
<body>
<main>

<h1>Graph-frequency band selector — {label}</h1>
<div class="source">{source_path}</div>

<div class="controls">

  <div class="row">
    <label for="low">Lowest selected frequency</label>
    <input id="low" type="range"
           min="0" max="{n_modes - 1}" value="2" step="1">
    <div class="value" id="lowText"></div>
  </div>

  <div class="row">
    <label for="high">Highest selected frequency</label>
    <input id="high" type="range"
           min="0" max="{n_modes - 1}"
           value="{min(8, n_modes - 1)}" step="1">
    <div class="value" id="highText"></div>
  </div>

  <div class="row">
    <label for="threshold">Keep points with strongest band energy</label>
    <input id="threshold" type="range"
           min="0" max="99.5" value="80" step="0.5">
    <div class="value" id="thresholdText"></div>
  </div>

  <div class="row">
    <label>Band presets</label>
    <div>
      <button id="singleBtn" type="button">Single mode</button>
      <button id="lowpassBtn" type="button">Low-pass</button>
      <button id="highpassBtn" type="button">High-pass</button>
      <button id="allBtn" type="button">All computed</button>
    </div>
    <button id="downloadBtn" type="button">
      Download selected XYZ
    </button>
  </div>

  <div class="stats">
    <span>Band: <b id="bandText"></b></span>
    <span>Selected: <b id="selectedText"></b></span>
    <span>Band global energy: <b id="bandEnergyText"></b></span>
    <span>Points: <b>{len(xyz):,}</b></span>
    <span>kNN: <b>{knn}</b></span>
    <span>σ: <b>{sigma:.6g}</b></span>
  </div>

</div>

<div class="grid">

  <div class="panel">
    <div id="cloud"></div>
  </div>

  <div>
    <div class="panel">
      <div id="spectrum"></div>
    </div>

    <div class="help">
      The two frequency sliders define a graph-frequency band
      [k<sub>low</sub>, k<sub>high</sub>]. For every point, the viewer sums
      the local XYZ spectral energy contributed by all modes in that band.
      The threshold slider keeps the strongest points. Move the frequency
      band through the spectrum to see where low-, middle-, and high-frequency
      geometry is spatially concentrated.
    </div>
  </div>

</div>

<script>
const DATA = {payload_json};

const xyz = DATA.xyz;
const eig = DATA.eigenvalues;
const globalEnergy = DATA.globalEnergy;
const prefix = DATA.prefixLocalEnergy; // prefix[q][point]

const lowSlider = document.getElementById("low");
const highSlider = document.getElementById("high");
const thresholdSlider = document.getElementById("threshold");

const lowText = document.getElementById("lowText");
const highText = document.getElementById("highText");
const thresholdText = document.getElementById("thresholdText");

const bandText = document.getElementById("bandText");
const selectedText = document.getElementById("selectedText");
const bandEnergyText = document.getElementById("bandEnergyText");

const singleBtn = document.getElementById("singleBtn");
const lowpassBtn = document.getElementById("lowpassBtn");
const highpassBtn = document.getElementById("highpassBtn");
const allBtn = document.getElementById("allBtn");
const downloadBtn = document.getElementById("downloadBtn");

const X = xyz.map(p => p[0]);
const Y = xyz.map(p => p[1]);
const Z = xyz.map(p => p[2]);

let selectedIndices = [];
let lastScores = [];

function percentile(values, p) {{
  if (!values.length) return 0;

  const a = values.slice().sort((x,y) => x-y);
  const f = (a.length - 1) * p / 100;
  const i0 = Math.floor(f);
  const i1 = Math.ceil(f);

  if (i0 === i1) return a[i0];

  const t = f - i0;
  return a[i0] * (1-t) + a[i1] * t;
}}

function enforceBand(changed) {{
  let lo = Number(lowSlider.value);
  let hi = Number(highSlider.value);

  if (lo > hi) {{
    if (changed === "low") {{
      highSlider.value = lo;
      hi = lo;
    }} else {{
      lowSlider.value = hi;
      lo = hi;
    }}
  }}

  return [lo, hi];
}}

function calculateBandScores(lo, hi) {{
  // prefix[q] contains modes [0, q-1].
  const before = prefix[lo];
  const after = prefix[hi + 1];

  const scores = new Array(xyz.length);

  for (let i=0; i<xyz.length; i++) {{
    scores[i] = Math.max(0, after[i] - before[i]);
  }}

  return scores;
}}

function calculateBandGlobalEnergy(lo, hi) {{
  let s = 0;
  for (let k=lo; k<=hi; k++) s += globalEnergy[k];
  return s;
}}

function renderSpectrum(lo, hi) {{
  const modes = eig.map((_,i) => i);

  const barColors = modes.map(k =>
    (k >= lo && k <= hi) ? "#f59e0b" : "#64748b"
  );

  const trace = {{
    type:"bar",
    x:modes,
    y:globalEnergy.map(v => 100*v),
    marker:{{color:barColors}},
    customdata:eig,
    hovertemplate:
      "mode k=%{{x}}<br>" +
      "lambda=%{{customdata:.6g}}<br>" +
      "XYZ energy=%{{y:.6f}}%" +
      "<extra></extra>"
  }};

  const layout = {{
    margin:{{l:58,r:16,t:40,b:48}},
    title:{{text:"Graph-frequency spectrum",font:{{size:15}}}},
    paper_bgcolor:"#181d23",
    plot_bgcolor:"#181d23",
    font:{{color:"#edf2f7"}},
    xaxis:{{
      title:"Graph mode k  → higher frequency",
      gridcolor:"#303945"
    }},
    yaxis:{{
      title:"Global XYZ spectral energy (%)",
      gridcolor:"#303945"
    }},
    showlegend:false,
    uirevision:"spectrum"
  }};

  Plotly.react(
    "spectrum",
    [trace],
    layout,
    {{responsive:true,displaylogo:false}}
  );

  const spectrumDiv = document.getElementById("spectrum");

  if (!spectrumDiv.__bandClickAttached) {{
    spectrumDiv.on("plotly_click", ev => {{
      if (!ev.points || !ev.points.length) return;
      const k = Number(ev.points[0].x);
      lowSlider.value = k;
      highSlider.value = k;
      render("low");
    }});
    spectrumDiv.__bandClickAttached = true;
  }}
}}

function render(changed="low") {{
  const [lo, hi] = enforceBand(changed);
  const thresholdP = Number(thresholdSlider.value);

  const scores = calculateBandScores(lo, hi);
  lastScores = scores;

  const scoreThreshold = percentile(scores, thresholdP);

  selectedIndices = [];
  for (let i=0; i<scores.length; i++) {{
    if (scores[i] >= scoreThreshold && scores[i] > 0) {{
      selectedIndices.push(i);
    }}
  }}

  const maxScore = Math.max(...scores, 1e-20);

  const main = {{
    type:"scatter3d",
    mode:"markers",
    x:X,
    y:Y,
    z:Z,
    customdata:scores.map((s,i) => [i,s]),
    marker:{{
      size:2.5,
      color:scores,
      cmin:0,
      cmax:maxScore,
      colorscale:"Turbo",
      showscale:true,
      opacity:0.82,
      colorbar:{{title:"band energy"}}
    }},
    hovertemplate:
      "point=%{{customdata[0]}}<br>" +
      "x=%{{x:.6g}}<br>" +
      "y=%{{y:.6g}}<br>" +
      "z=%{{z:.6g}}<br>" +
      "band energy=%{{customdata[1]:.6g}}" +
      "<extra></extra>"
  }};

  const sx=[], sy=[], sz=[];

  for (const idx of selectedIndices) {{
    sx.push(xyz[idx][0]);
    sy.push(xyz[idx][1]);
    sz.push(xyz[idx][2]);
  }}

  const selected = {{
    type:"scatter3d",
    mode:"markers",
    x:sx,
    y:sy,
    z:sz,
    marker:{{
      size:4.2,
      color:"#ffffff",
      opacity:0.72
    }},
    hoverinfo:"skip",
    showlegend:false
  }};

  const layout = {{
    margin:{{l:0,r:0,t:40,b:0}},
    title:{{
      text:`Spatial energy in modes ${{lo}}…${{hi}}`,
      font:{{size:15}}
    }},
    paper_bgcolor:"#181d23",
    plot_bgcolor:"#181d23",
    font:{{color:"#edf2f7"}},
    scene:{{
      aspectmode:"data",
      bgcolor:"#181d23",
      xaxis:{{title:"X",gridcolor:"#303945"}},
      yaxis:{{title:"Y",gridcolor:"#303945"}},
      zaxis:{{title:"Z",gridcolor:"#303945"}}
    }},
    uirevision:"keep-camera"
  }};

  Plotly.react(
    "cloud",
    [main,selected],
    layout,
    {{responsive:true,displaylogo:false}}
  );

  lowText.textContent =
    `k=${{lo}}, λ=${{Number(eig[lo]).toExponential(4)}}`;

  highText.textContent =
    `k=${{hi}}, λ=${{Number(eig[hi]).toExponential(4)}}`;

  thresholdText.textContent =
    `top ${{(100-thresholdP).toFixed(1)}}%`;

  bandText.textContent = `${{lo}} … ${{hi}}`;

  selectedText.textContent =
    `${{selectedIndices.length.toLocaleString()}} / ` +
    `${{xyz.length.toLocaleString()}}`;

  bandEnergyText.textContent =
    `${{(100*calculateBandGlobalEnergy(lo,hi)).toFixed(5)}}%`;

  renderSpectrum(lo,hi);
}}

lowSlider.addEventListener("input", () => render("low"));
highSlider.addEventListener("input", () => render("high"));
thresholdSlider.addEventListener("input", () => render());

singleBtn.addEventListener("click", () => {{
  const k = Number(highSlider.value);
  lowSlider.value = k;
  highSlider.value = k;
  render();
}});

lowpassBtn.addEventListener("click", () => {{
  lowSlider.value = 0;
  render("low");
}});

highpassBtn.addEventListener("click", () => {{
  highSlider.value = eig.length - 1;
  render("high");
}});

allBtn.addEventListener("click", () => {{
  lowSlider.value = 0;
  highSlider.value = eig.length - 1;
  render();
}});

downloadBtn.addEventListener("click", () => {{
  if (!selectedIndices.length) return;

  const lo = Number(lowSlider.value);
  const hi = Number(highSlider.value);

  let csv = "index,x,y,z,band_energy\\n";

  for (const i of selectedIndices) {{
    csv += `${{i}},${{xyz[i][0]}},${{xyz[i][1]}},${{xyz[i][2]}},${{lastScores[i]}}\\n`;
  }}

  const blob = new Blob([csv],{{type:"text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `selected_frequency_band_${{lo}}_${{hi}}.csv`;

  document.body.appendChild(a);
  a.click();
  a.remove();

  URL.revokeObjectURL(url);
}});

render();
</script>
</main>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")


# =============================================================================
# Processing
# =============================================================================

def process_cloud(path_str, label, args):
    source = Path(path_str).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")

    outdir = Path(args.output).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / f"{label}_graph_spectrum.npz"
    html_path = outdir / f"{label}_graph_frequency.html"

    tqdm.write("")
    tqdm.write(f"=== {label} ===")
    tqdm.write(f"source : {source}")
    tqdm.write(f"NPZ    : {npz_path}")
    tqdm.write(f"HTML   : {html_path}")

    with tqdm(
        total=7,
        desc=label,
        unit="stage",
        dynamic_ncols=True,
    ) as stage:

        stage.set_postfix_str("load PLY")
        pcd, xyz = load_ply_xyz(str(source))
        original_count = len(xyz)
        stage.update(1)

        tqdm.write(f"  loaded {original_count:,} points")

        stage.set_postfix_str("preprocess")

        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)
            xyz = np.asarray(pcd.points, dtype=np.float64)
            tqdm.write(
                f"  voxel: {original_count:,} -> {len(xyz):,}"
            )

        xyz, source_indices = deterministic_subsample(
            xyz,
            args.max_points,
            args.seed,
        )

        if len(xyz) != original_count:
            tqdm.write(f"  using {len(xyz):,} points")

        stage.update(1)

        stage.set_postfix_str("k-NN graph")
        W, sigma = build_knn_graph(
            xyz,
            k=args.knn,
            sigma=args.sigma,
            chunk_size=args.knn_chunk,
            desc=f"{label}: k-NN",
        )
        stage.update(1)

        n_components, cc_labels = csgraph.connected_components(
            W,
            directed=False,
        )

        tqdm.write(f"  graph sigma = {sigma:.8g}")
        tqdm.write(f"  graph edges = {W.nnz // 2:,}")
        tqdm.write(f"  connected components = {n_components}")

        if n_components > 1:
            sizes = np.bincount(cc_labels)
            tqdm.write(
                "  WARNING: disconnected k-NN graph; "
                f"largest components={np.sort(sizes)[::-1][:8].tolist()}"
            )

        stage.set_postfix_str(f"eigensolve {args.modes} modes")
        tqdm.write(
            f"  computing {args.modes} lowest graph frequencies..."
        )

        eigvals, eigvecs = graph_spectrum(
            W,
            n_modes=args.modes,
            normalized=not args.unnormalized,
            tol=args.eig_tol,
        )
        stage.update(1)

        stage.set_postfix_str("spectral XYZ energy")
        (
            coeff,
            global_energy,
            global_energy_norm,
            cumulative_global,
            local_energy_norm,
            local_energy_prefix,
        ) = compute_geometry_spectrum(
            xyz,
            eigvecs,
        )
        stage.update(1)

        tqdm.write("  first frequencies:")
        for k in range(min(12, len(eigvals))):
            tqdm.write(
                f"    k={k:3d}  "
                f"lambda={eigvals[k]:.9g}  "
                f"XYZ_energy={100*global_energy_norm[k]:.6f}%"
            )

        stage.set_postfix_str("save NPZ")
        np.savez_compressed(
            npz_path,
            xyz=xyz.astype(np.float32),
            source_indices=source_indices,
            eigenvalues=eigvals.astype(np.float64),
            eigenvectors=eigvecs.astype(np.float32),
            graph_fourier_xyz=coeff.astype(np.float64),
            global_geometry_energy=global_energy.astype(np.float64),
            global_geometry_energy_normalized=global_energy_norm.astype(np.float64),
            cumulative_global_geometry_energy=cumulative_global.astype(np.float64),
            local_geometry_energy_normalized=local_energy_norm.astype(np.float32),
            knn=np.asarray(args.knn),
            sigma=np.asarray(sigma),
            source_path=np.asarray(str(source)),
            output_label=np.asarray(label),
        )
        stage.update(1)

        stage.set_postfix_str("save interactive HTML")
        write_html(
            path=html_path,
            label=label,
            source_path=source,
            xyz=xyz,
            eigvals=eigvals,
            global_energy_norm=global_energy_norm,
            cumulative_global=cumulative_global,
            local_energy_prefix=local_energy_prefix,
            knn=args.knn,
            sigma=sigma,
            original_count=original_count,
        )
        stage.update(1)

    return npz_path, html_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Graph-frequency decomposition of independent PLY point clouds "
            "with an interactive frequency-band point selector."
        ),
    )

    parser.add_argument(
        "ply",
        nargs="*",
        help=(
            "Input PLYs. If omitted, the two storage_45135 endpoint PLYs "
            "hard-coded in this script are used."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory.",
    )

    parser.add_argument(
        "--knn",
        type=int,
        default=20,
        help="Spatial k-NN graph degree.",
    )

    parser.add_argument(
        "--modes",
        type=int,
        default=100,
        help="Number of lowest graph-frequency modes.",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum points per cloud; 0 disables cap.",
    )

    parser.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="Optional voxel size before graph construction.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Graph Gaussian scale; auto=median k-NN edge distance.",
    )

    parser.add_argument(
        "--knn-chunk",
        type=int,
        default=8192,
        help="Points per cKDTree query chunk for tqdm updates.",
    )

    parser.add_argument(
        "--eig-tol",
        type=float,
        default=1e-5,
        help="ARPACK eigsh tolerance.",
    )

    parser.add_argument(
        "--unnormalized",
        action="store_true",
        help="Use L=D-W instead of normalized Laplacian.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Subsampling random seed.",
    )

    args = parser.parse_args()

    inputs = args.ply if args.ply else DEFAULT_INPUTS

    labels = make_unique_labels(inputs)

    outdir = Path(args.output).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Compute all output paths up front and explicitly verify no two inputs
    # can overwrite each other.
    expected = []
    for label in labels:
        expected.extend([
            outdir / f"{label}_graph_spectrum.npz",
            outdir / f"{label}_graph_frequency.html",
        ])

    if len(expected) != len(set(expected)):
        raise RuntimeError(
            "Output filename collision detected before processing:\n" +
            "\n".join(str(p) for p in expected)
        )

    tqdm.write("Input -> output mapping:")
    for src, label in zip(inputs, labels):
        tqdm.write(f"  {label}")
        tqdm.write(f"    <- {Path(src).expanduser()}")
        tqdm.write(f"    -> {outdir / (label + '_graph_frequency.html')}")

    results = []

    for src, label in tqdm(
        list(zip(inputs, labels)),
        desc="Point clouds",
        unit="cloud",
        dynamic_ncols=True,
    ):
        results.append(
            process_cloud(
                src,
                label,
                args,
            )
        )

    tqdm.write("")
    tqdm.write("DONE")

    for label, (npz_path, html_path) in zip(labels, results):
        tqdm.write(f"  [{label}]")
        tqdm.write(f"    {npz_path}")
        tqdm.write(f"    {html_path}")


if __name__ == "__main__":
    main()
