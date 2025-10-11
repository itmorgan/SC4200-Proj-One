#!/usr/bin/env python3
#!/usr/bin/env python3
"""
DBSCAN on Yeast & Human PPI (no node2vec/gensim)
------------------------------------------------
- Reads yeast.edgelist and PP-Pathways_ppi.csv
- Builds SVD embeddings from adjacency (fast & dependency-light)
- Sweeps DBSCAN hyperparams, picks best by silhouette
- Saves labeled embeddings and grid results as CSVs
"""

import argparse
import os
import sys
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# --------- Utils ---------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def check_file_exists(path: str, desc: str):
    if not os.path.isfile(path):
        eprint(f"[ERROR] {desc} not found at: {path}")
        sys.exit(1)

# --------- Load graphs ---------
def load_yeast_graph(path: str) -> nx.Graph:
    G = nx.read_edgelist(path)
    print(f"[Yeast] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G

def load_human_graph(path: str) -> nx.Graph:
    df = pd.read_csv(path)
    # Try common PPI column names first; else use the first two columns
    candidates = [
        ("protein1", "protein2"),
        ("Protein1", "Protein2"),
        ("prot1", "prot2"),
        ("u", "v"),
        ("source", "target"),
        ("from", "to"),
    ]
    ucol = vcol = None
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            ucol, vcol = a, b
            break
    if ucol is None:
        # Fallback — first two columns
        ucol, vcol = df.columns[:2]
        print(f"[Human] Using columns '{ucol}' and '{vcol}' as edges (adjust if needed).")
    edges = list(zip(df[ucol].astype(str), df[vcol].astype(str)))
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"[Human] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G

# --------- Embeddings via SVD ---------
def svd_embeddings(
    G: nx.Graph,
    dimensions: int = 64,
    random_state: int = 0,
    keep_lcc: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Returns:
      emb_df: DataFrame [n_nodes x (dimensions + node)]
      Xs: scaled embedding for clustering
      nodes: node order
    """
    if keep_lcc and not nx.is_empty(G):
        if not nx.is_connected(G):
            lcc_nodes = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc_nodes).copy()
            print(f"[SVD] Kept largest component: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes after preprocessing.")

    nodes = list(G.nodes())
    # Sparse adjacency as CSR
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=np.float32, format="csr")

    d = min(dimensions, min(A.shape) - 1)
    d = max(d, 2)  # ensure >=2
    svd = TruncatedSVD(n_components=d, random_state=random_state)
    X = svd.fit_transform(A)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    emb_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    emb_df["node"] = nodes
    return emb_df, Xs, nodes

# --------- K-distance elbow (optional) ---------
def k_distance_plot(Xs: np.ndarray, k: int = 10, title: str = "k-distance (elbow ~ eps)"):
    nbrs = NearestNeighbors(n_neighbors=k).fit(Xs)
    dists, _ = nbrs.kneighbors(Xs)
    kd = np.sort(dists[:, -1])
    plt.figure()
    plt.plot(kd)
    plt.title(title)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}-th NN")
    plt.tight_layout()
    plt.show()

# --------- DBSCAN grid ---------
def run_dbscan_grid(
    Xs: np.ndarray,
    eps_list: Tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 2.0, 3.0),
    min_samples_list: Tuple[int, ...] = (5, 10, 20, 30)
) -> Tuple[dict, pd.DataFrame]:
    results = []
    best = {"sil": -np.inf, "labels": None, "params": None, "n_clusters": 0, "noise_ratio": None}
    for eps in eps_list:
        for ms in min_samples_list:
            model = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels = model.fit_predict(Xs)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise = float(np.mean(labels == -1))
            if n_clusters >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        sil = float(silhouette_score(Xs, labels))
                    except Exception:
                        sil = np.nan
            else:
                sil = np.nan
            results.append((eps, ms, n_clusters, noise, sil))
            if not np.isnan(sil) and sil > best["sil"]:
                best.update({
                    "sil": sil,
                    "labels": labels,
                    "params": (eps, ms),
                    "n_clusters": n_clusters,
                    "noise_ratio": noise
                })
    res_df = pd.DataFrame(results, columns=["eps", "min_samples", "n_clusters", "noise_ratio", "silhouette"])\
        .sort_values(["silhouette", "n_clusters"], ascending=[False, False]).reset_index(drop=True)
    return best, res_df

# --------- PCA scatter (optional) ---------
def pca_scatter(Xs: np.ndarray, labels: np.ndarray, title: str):
    if labels is None or len(set(labels)) <= 1:
        print(f"[Plot] {title}: skipping (no meaningful clusters).")
        return
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(Xs)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], s=8, alpha=0.85, c=labels)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser(description="DBSCAN on Yeast & Human PPI using SVD embeddings")
    ap.add_argument("--yeast", default="yeast.edgelist", help="Path to yeast edgelist file")
    ap.add_argument("--human", default="PP-Pathways_ppi.csv", help="Path to human PPI CSV")
    ap.add_argument("--dims", type=int, default=64, help="Embedding dimension (SVD)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--eps", type=str, default="0.8,1.0,1.2,1.5,2.0,3.0", help="Comma-separated eps values")
    ap.add_argument("--min_samples", type=str, default="5,10,20,30", help="Comma-separated min_samples values")
    ap.add_argument("--plots", action="store_true", help="Show PCA cluster plots")
    ap.add_argument("--kdist", action="store_true", help="Show k-distance elbow plots (k=10)")
    ap.add_argument("--outdir", default=".", help="Directory to save CSV outputs")
    args = ap.parse_args()

    check_file_exists(args.yeast, "Yeast edgelist")
    check_file_exists(args.human, "Human PPI CSV")

    eps_list = tuple(float(x.strip()) for x in args.eps.split(",") if x.strip())
    ms_list = tuple(int(x.strip()) for x in args.min_samples.split(",") if x.strip())

    # Load graphs
    G_yeast = load_yeast_graph(args.yeast)
    G_human = load_human_graph(args.human)

    # Embeddings
    embY_df, XsY, nodesY = svd_embeddings(G_yeast, dimensions=args.dims, random_state=args.seed)
    embH_df, XsH, nodesH = svd_embeddings(G_human, dimensions=args.dims, random_state=args.seed)
    print(f"[Embeddings] Yeast: {embY_df.shape}, Human: {embH_df.shape}")

    # Optional k-distance elbow
    if args.kdist:
        k_distance_plot(XsY, k=10, title="Yeast: k-distance (k=10)")
        k_distance_plot(XsH, k=10, title="Human: k-distance (k=10)")

    # DBSCAN grid
    bestY, gridY = run_dbscan_grid(XsY, eps_list=eps_list, min_samples_list=ms_list)
    bestH, gridH = run_dbscan_grid(XsH, eps_list=eps_list, min_samples_list=ms_list)

    # Reports
    def report(name, best, grid):
        print(f"\n=== {name}: DBSCAN results ===")
        if best["params"] is None:
            print("No configuration produced >= 2 clusters. Try larger eps or smaller min_samples.")
        else:
            eps, ms = best["params"]
            print(f"Best params: eps={eps}, min_samples={ms}")
            print(f"Silhouette: {best['sil']:.4f}")
            print(f"#Clusters (excl. noise): {best['n_clusters']}")
            print(f"Noise ratio: {best['noise_ratio']:.3f}")
        print("\nTop candidates:")
        print(grid.head(10).to_string(index=False))

    report("Yeast", bestY, gridY)
    report("Human", bestH, gridH)

    # Attach labels
    embY_df["dbscan_label"] = bestY["labels"] if bestY["labels"] is not None else -1
    embH_df["dbscan_label"] = bestH["labels"] if bestH["labels"] is not None else -1

    # Plots
    if args.plots:
        pca_scatter(XsY, bestY["labels"], "Yeast: DBSCAN clusters (PCA)")
        pca_scatter(XsH, bestH["labels"], "Human: DBSCAN clusters (PCA)")

    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    out_ye = os.path.join(args.outdir, "yeast_embeddings_dbscan.csv")
    out_hu = os.path.join(args.outdir, "human_embeddings_dbscan.csv")
    out_grid_y = os.path.join(args.outdir, "dbscan_grid_yeast.csv")
    out_grid_h = os.path.join(args.outdir, "dbscan_grid_human.csv")

    embY_df.to_csv(out_ye, index=False)
    embH_df.to_csv(out_hu, index=False)
    gridY.to_csv(out_grid_y, index=False)
    gridH.to_csv(out_grid_h, index=False)

    print(f"\nSaved:\n  {out_ye}\n  {out_hu}\n  {out_grid_y}\n  {out_grid_h}")

if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        # Running inside Jupyter or IPython — call main() manually with defaults
        sys.argv = ["", "--yeast", "yeast.edgelist", "--human", "bio-pathways-network.csv"]
    try:
        main()
    except KeyboardInterrupt:
        eprint("\n[Interrupted]")
        sys.exit(130)
    except Exception as e:
        eprint(f"[FATAL] {e}")
        raise

"""
DBSCAN Reporting & Visualizations
---------------------------------
Loads:
  - yeast_embeddings_dbscan.csv
  - human_embeddings_dbscan.csv
  - dbscan_grid_yeast.csv
  - dbscan_grid_human.csv

Produces (in reports_dbscan/):
  - dbscan_performance_summary.csv
  - yeast_dbscan_pca.png
  - human_dbscan_pca.png
  - yeast_dbscan_param_heatmap.png
  - human_dbscan_param_heatmap.png
  - yeast_dbscan_cluster_sizes.png
  - human_dbscan_cluster_sizes.png
  - yeast_dbscan_param_curves.png  (fallback curves if heatmap pivot fails)
  - human_dbscan_param_curves.png  (fallback curves if heatmap pivot fails)

These outputs are ready to compare later against K-Means and GCN.
"""

import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# -----------------------
# Config (edit if needed)
# -----------------------
IN_YEAST_EMB = "yeast_embeddings_dbscan.csv"
IN_HUMAN_EMB = "human_embeddings_dbscan.csv"
IN_YEAST_GRID = "dbscan_grid_yeast.csv"
IN_HUMAN_GRID = "dbscan_grid_human.csv"

OUTDIR = "reports_dbscan"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def feature_cols(df: pd.DataFrame):
    """Return columns named f0, f1, ..."""
    return [c for c in df.columns if re.fullmatch(r"f\d+", c)]

def safe_silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int = 5000) -> float:
    """Silhouette with optional subsampling to avoid O(N^2) cost on very large sets."""
    uniq = set(labels)
    if len(uniq) < 2:
        return float("nan")
    n = len(X)
    if n > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        X_sub = X[idx]
        y_sub = labels[idx]
        return float(silhouette_score(X_sub, y_sub))
    return float(silhouette_score(X, labels))

def compute_metrics(emb_df: pd.DataFrame) -> Dict[str, float]:
    """Compute standard external metrics for clustering quality."""
    X = emb_df[feature_cols(emb_df)].values
    labels = emb_df["dbscan_label"].values
    metrics = {}
    if len(set(labels)) <= 1:
        return {
            "silhouette": float("nan"),
            "CH_index": float("nan"),
            "DB_index": float("nan"),
            "noise_ratio": float(np.mean(labels == -1)),
            "n_clusters": 0,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sil = safe_silhouette(X, labels)
        ch = float(calinski_harabasz_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float(np.mean(labels == -1))
    return {
        "silhouette": sil,
        "CH_index": ch,
        "DB_index": db,
        "noise_ratio": noise_ratio,
        "n_clusters": n_clusters,
    }

def pca_scatter(emb_df: pd.DataFrame, title: str, out_png: str):
    """2D PCA scatter colored by DBSCAN label (-1 = noise)."""
    X = emb_df[feature_cols(emb_df)].values
    y = emb_df["dbscan_label"].values
    if len(np.unique(y)) <= 1:
        print(f"[Plot] {title}: skipped (degenerate labels).")
        return
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    # Map labels to integers for a stable colormap
    uniq = np.unique(y)
    # Keep -1 as last color if present
    uniq_sorted = np.array(sorted([u for u in uniq if u != -1]) + ([-1] if -1 in uniq else []))
    label_to_idx = {lab: i for i, lab in enumerate(uniq_sorted)}
    colors = [label_to_idx[lab] for lab in y]

    sc = plt.scatter(X2[:, 0], X2[:, 1], c=colors, s=8, alpha=0.85)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # Simple colorbar with tick labels matching cluster IDs
    cbar = plt.colorbar(sc, ticks=list(range(len(uniq_sorted))))
    cbar.ax.set_yticklabels([str(int(l)) for l in uniq_sorted])
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def cluster_size_bar(labels: np.ndarray, title: str, out_png: str):
    """Bar chart of cluster sizes (including noise as -1 if present)."""
    vals, counts = np.unique(labels, return_counts=True)
    # sort: normal clusters desc, keep -1 at end
    order = [v for v in vals if v != -1]
    order = sorted(order, key=lambda v: counts[np.where(vals == v)[0][0]], reverse=True)
    if -1 in vals:
        order.append(-1)
    sizes = [counts[np.where(vals == v)[0][0]] for v in order]

    plt.figure(figsize=(8, 5))
    plt.bar([str(int(v)) for v in order], sizes)
    plt.title(title)
    plt.xlabel("Cluster label")
    plt.ylabel("# Nodes")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def heatmap_from_grid(grid_df: pd.DataFrame, title: str, out_png_heat: str, out_png_curves: str):
    """
    Try to plot a heatmap silhouette over (eps, min_samples). If pivot fails (e.g., floats that don't pivot well),
    fallback to line curves of silhouette vs eps grouped by min_samples.
    """
    # Clean numeric
    g = grid_df.copy()
    for col in ["eps", "min_samples", "silhouette"]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=["eps", "min_samples", "silhouette"])

    # Try pivot-based heatmap
    try:
        pivot = g.pivot_table(index="eps", columns="min_samples", values="silhouette", aggfunc="max")
        # sort axes numerically
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
        plt.figure(figsize=(7, 5))
        im = plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.title(title + " — DBSCAN silhouette heatmap")
        plt.xlabel("min_samples")
        plt.ylabel("eps")
        # ticks
        plt.xticks(range(pivot.shape[1]), [str(int(c)) for c in pivot.columns])
        # round eps nicely
        yticklabels = [("{:.2f}".format(r)).rstrip("0").rstrip(".") for r in pivot.index]
        plt.yticks(range(pivot.shape[0]), yticklabels)
        cbar = plt.colorbar(im)
        cbar.set_label("silhouette", rotation=270, labelpad=12)
        plt.tight_layout()
        plt.savefig(out_png_heat, dpi=160)
        plt.close()
        return
    except Exception as e:
        print(f"[Heatmap] pivot failed ({e}); falling back to curves.")

    # Fallback: curves of silhouette vs eps per min_samples
    mins = sorted(g["min_samples"].unique())
    plt.figure(figsize=(7, 5))
    for ms in mins:
        sub = g[g["min_samples"] == ms].sort_values("eps")
        plt.plot(sub["eps"], sub["silhouette"], marker="o", label=f"min_samples={int(ms)}")
    plt.title(title + " — DBSCAN silhouette vs eps")
    plt.xlabel("eps")
    plt.ylabel("silhouette")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_curves, dpi=160)
    plt.close()

def load_embeddings_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    # Ensure required pieces exist
    fcols = feature_cols(df)
    if not fcols:
        raise ValueError(f"No embedding feature columns found in {path}")
    if "dbscan_label" not in df.columns:
        raise ValueError(f"'dbscan_label' column not found in {path}")
    return df

def analyze_one(dataset_name: str, emb_path: str, grid_path: str) -> Dict[str, float]:
    print(f"\n=== {dataset_name} ===")
    emb_df = load_embeddings_csv(emb_path)
    X = emb_df[feature_cols(emb_df)].values
    y = emb_df["dbscan_label"].values

    # Metrics
    M = compute_metrics(emb_df)
    print(
        f"DBSCAN: clusters={M['n_clusters']} | noise={M['noise_ratio']:.3f} | "
        f"sil={M['silhouette']:.4f} | CH={M['CH_index']:.2f} | DB={M['DB_index']:.3f}"
    )

    # PCA scatter
    pca_scatter(
        emb_df,
        f"{dataset_name} — DBSCAN (PCA)",
        os.path.join(OUTDIR, f"{dataset_name.lower()}_dbscan_pca.png")
    )

    # Cluster sizes
    cluster_size_bar(
        y,
        f"{dataset_name} — DBSCAN cluster sizes",
        os.path.join(OUTDIR, f"{dataset_name.lower()}_dbscan_cluster_sizes.png")
    )

    # Parameter heatmap/curves
    if os.path.isfile(grid_path):
        grid_df = pd.read_csv(grid_path)
        heatmap_from_grid(
            grid_df,
            dataset_name,
            os.path.join(OUTDIR, f"{dataset_name.lower()}_dbscan_param_heatmap.png"),
            os.path.join(OUTDIR, f"{dataset_name.lower()}_dbscan_param_curves.png"),
        )
    else:
        print(f"[Warn] grid file not found for {dataset_name}: {grid_path}")

    # Return row for summary CSV
    return {
        "dataset": dataset_name,
        "clusters": M["n_clusters"],
        "noise_ratio": M["noise_ratio"],
        "silhouette": M["silhouette"],
        "Calinski_Harabasz": M["CH_index"],
        "Davies_Bouldin": M["DB_index"],
    }

# -----------------------
# Run for both datasets
# -----------------------
if __name__ == "__main__":
    rows = []
    rows.append(analyze_one("Yeast", IN_YEAST_EMB, IN_YEAST_GRID))
    rows.append(analyze_one("Human", IN_HUMAN_EMB, IN_HUMAN_GRID))

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(OUTDIR, "dbscan_performance_summary.csv")
    summary.to_csv(out_csv, index=False)
    print("\nSaved summary:", out_csv)
    print("Artifacts written to:", os.path.abspath(OUTDIR))
