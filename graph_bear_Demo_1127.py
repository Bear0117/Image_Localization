import argparse
import json
import math
from pathlib import Path
import faiss
import networkx as nx
import numpy as np
import csv

import os
from networkx.algorithms.similarity import optimize_edit_paths
from typing import Optional, List, Dict, Tuple, Sequence, Mapping, Any
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
import open3d as o3d
from construct_real_graph import real_main

# Test
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
DELTA = 1.0
EPSILON = 1.0
LAMBDA_NODE = 1.0
LAMBDA_EDGE = 0.5
N_sim = 1.0
E_sim = 1.0 #如果測試 elevator 要設成 0


# === Layout (2x2) 固定參數 ===
IMG_W, IMG_H = 1920, 1080
LAYOUT_ROWS, LAYOUT_COLS = 2, 2
# 類別順序（固定 9 類）
STRUCTURAL_CLASSES = ["Walls","Columns","Beams","Ceilings","Floors","Doors","Windows","Pipe","Cable_Tray"]
CLASS_COLORS: Dict[str, Tuple[int,int,int]] = {
    "Walls": (60, 60, 255),
    "Columns": (128, 128, 0),
    "Beams": (0, 165, 255),
    "Ceilings": (255, 0, 255),
    "Floors": (0, 255, 255),
    "Doors": (0, 200, 0),
    "Windows": (255, 255, 0),
    "Pipe": (200, 0, 100),
    "Cable_Tray": (180, 180, 180),
}

LAYOUT_TAU = 1.0
LAYOUT_WEIGHT = 1.0
STAT_PER_CLASS = 5

# ---------------- Camera Intrinsics of real images ----------------
FX_REAL = 1649.545
FY_REAL = 1649.719
CX_REAL = 1200.0
CY_REAL = 673.5

# ---------------- Camera Intrinsics of synthetic images ----------------
FX_SYN = 2198.998
FY_SYN = 1695.137
CX_SYN = 960.0
CY_SYN = 540.0

# ==== Sequence (x/y) utilities for LCS-XY ====
def build_seq_xy(g, class_order, exclude_set=None, intrinsics: str = "auto"):
    """
    由單一 graph 產生兩條「以影像座標排序」的類別序列：
      - seq_x: 依影像 u 由小到大排序後的 class_id 序列
      - seq_y: 依影像 v 由小到大排序後的 class_id 序列

    intrinsics:
      "real" | "syn" | "auto"
      - auto：依 g.graph["image"] / g.graph["img_path"] 判斷（.png 或含 "_rgb_image" 視為 syn）
    """
    # 0) 準備：類別 → id 對照與「是否排除」
    cls2id = {c: i for i, c in enumerate(class_order)}
    def _keep(cat: str) -> bool:
        return (cat in cls2id) and not (exclude_set and cat in exclude_set)

    # 1) Intrinsic Selection
    if intrinsics == "syn":
        fx, fy, cx, cy = FX_SYN, FY_SYN, CX_SYN, CY_SYN
    elif intrinsics == "real":
        fx, fy, cx, cy = FX_REAL, FY_REAL, CX_REAL, CY_REAL
    else:
        img_name = str(g.graph.get("image", g.graph.get("img_path", ""))).lower()
        is_syn = ("_rgb_image" in img_name) or img_name.endswith(".png")
        fx, fy, cx, cy = (FX_SYN, FY_SYN, CX_SYN, CY_SYN) if is_syn else (FX_REAL, FY_REAL, CX_REAL, CY_REAL)

    # 2) 先收集 X/Y 範圍，供 Z 無效時的 fallback 正規化
    xs, ys = [], []
    for _, d in g.nodes(data=True):
        if ("cx" in d) and ("cy" in d):
            xs.append(float(d["cx"]))
            ys.append(float(d["cy"]))
    min_x = min(xs) if xs else 0.0
    max_x = max(xs) if xs else 1.0
    min_y = min(ys) if ys else 0.0
    max_y = max(ys) if ys else 1.0
    rx = max(max_x - min_x, 1e-6)
    ry = max(max_y - min_y, 1e-6)

    # 3) 投影到 (u,v)，對 Z 無效者以 (cx,cy) 做 min–max fallback
    nodes = []  # (u, v, class_id)
    for _, d in g.nodes(data=True):
        cat = d.get("category", None)
        if not _keep(cat):
            continue
        if not all(k in d for k in ("cx", "cy")):
            continue

        X = float(d["cx"]); Y = float(d["cy"])
        Z = float(d["cz"]) if ("cz" in d) else float("nan")

        uvt = None
        if math.isfinite(Z) and (Z > 1e-8):
            # 使用已存在的投影定義：u = fx*(X/Z)+cx, v = fy*(Y/Z)+cy
            uvt = _project_uv((X, Y, Z), fx, fy, cx, cy)

        if uvt is None:
            # 無法投影（Z 無效）：退回以 (cx,cy) 的相對位置排序
            u = (X - min_x) / rx
            v = (Y - min_y) / ry
        else:
            u, v = float(uvt[0]), float(uvt[1])

        nodes.append((u, v, cls2id[cat]))

    # 4) 依 u、v 產生兩條序列（與原 ties 規則一致）
    nodes_x = sorted(nodes, key=lambda t: (t[0], t[1]))  # u↑，同 u 用 v↑
    nodes_y = sorted(nodes, key=lambda t: (t[1], t[0]))  # v↑，同 v 用 u↑

    seq_x = [cid for _, _, cid in nodes_x]
    seq_y = [cid for _, _, cid in nodes_y]
    return seq_x, seq_y

def _lcs_len(a, b):
    """
    標準 LCS 長度，O(n*m)；n,m 一般不大，足夠用。
    回傳整數 L
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # 2-row DP 節省記憶體
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev, curr = curr, prev
    return prev[m]

def seq_lcs_similarity_xy(seqxA, seqxB, seqyA, seqyB, wx=0.5, wy=0.5):
    """
    雙軸 LCS 相似度（預設 wx=wy=0.5；不提供 CLI 參數，寫死預設）
      Sx = 2*Lx/(nx+ny), Sy = 2*Ly/(mx+my), S = wx*Sx + wy*Sy
    回傳：S, Sx, Sy, 距離 D=1-S
    """
    nx, ny = len(seqxA), len(seqxB)
    mx, my = len(seqyA), len(seqyB)
    Lx = _lcs_len(seqxA, seqxB) if nx > 0 and ny > 0 else 0
    Ly = _lcs_len(seqyA, seqyB) if mx > 0 and my > 0 else 0
    Sx = (2.0 * Lx / (nx + ny)) if (nx + ny) > 0 else 0.0
    Sy = (2.0 * Ly / (mx + my)) if (mx + my) > 0 else 0.0
    S  = wx * Sx + wy * Sy
    D  = 1.0 - S
    return float(S), float(Sx), float(Sy), float(D)

def build_seq_xy_batch(graphs, class_order, exclude_classes_str="", intrinsics: str = "auto"):
    """
    將多個 graphs 轉為 [(seq_x, seq_y), ...]
    exclude_classes_str: 預設空字串 => 不排除；若未來想排除可傳逗號字串，但這版不接 CLI。
    """
    exclude = set([s.strip() for s in exclude_classes_str.split(",") if s.strip()]) if exclude_classes_str else None
    out = []
    for g in graphs:
        sx, sy = build_seq_xy(g, class_order, exclude_set=exclude, intrinsics=intrinsics)
        out.append((sx, sy))
    return out

# ---------------- Layout helpers (2x2) ----------------
def _project_uv(center_xyz, fx: float, fy: float, cx: float, cy: float) -> Optional[tuple]:
    X, Y, Z = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    if not math.isfinite(Z) or Z <= 1e-8:
        return None
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return (u, v)

# [A-2] 小工具：canonical 命名 + 讀 MI
def _canon_name(s: str) -> str:
    b = Path(s).name.lower()
    if "." in b:
        b = b.rsplit(".", 1)[0]
    if b.endswith("_rgb_image"):
        b = b[:-len("_rgb_image")]
    return b


# ---------------- Wireframe helpers (JSON -> features) ----------------
def _get_fname(entry: dict) -> str:
    v = entry.get("filename")
    if isinstance(v, (list, tuple)): 
        return v[0]
    return v

def load_wire_json(path: Path) -> List[dict]:
    return json.loads(Path(path).read_text())

def preprocess_lines_np(entry: dict, H: int, W: int, score_th: float) -> np.ndarray:
    """Filter by score_th and resize to (H,W). Return Nx4 float32 (x1,y1,x2,y2)."""
    lines = np.asarray(entry.get("lines_pred", []), np.float32)
    if lines.size == 0:
        return lines.reshape(0, 4)

    # optional score filter
    scores = np.asarray(entry.get("lines_score", []), np.float32).reshape(-1)
    if scores.size == lines.shape[0]:
        keep = (scores >= float(score_th))
        lines = lines[keep]

    # resize to target (H,W)
    src_h = int(entry.get("height", H))
    src_w = int(entry.get("width",  W))
    if (src_h, src_w) != (H, W) and lines.size > 0:
        sy, sx = float(H) / max(1, src_h), float(W) / max(1, src_w)
        lines = lines.copy()
        lines[:, [0, 2]] *= sx
        lines[:, [1, 3]] *= sy
    return lines

def compute_cell_features_np(lines: np.ndarray, H: int, W: int, rows: int, cols: int, num_bins: int) -> np.ndarray:
    """Grid orientation hist: (rows*cols*num_bins,) with per-cell L1 normalization."""
    feats = np.zeros((rows * cols, num_bins), dtype=np.float32)
    if lines.size == 0:
        return feats.reshape(-1)

    cell_h, cell_w = float(H) / rows, float(W) / cols

    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-3:
            continue

        # angle in [0,180)
        ang = (np.degrees(np.arctan2(dy, dx)) % 180.0)
        b = int(ang / (180.0 / max(1, num_bins)))
        b = max(0, min(num_bins - 1, b))

        # sample along the segment (every ~10 px, at least 1 step)
        npts = max(int(length // 10), 1)
        xs = np.linspace(x1, x2, npts + 1)
        ys = np.linspace(y1, y2, npts + 1)

        for i in range(npts):
            mx = (xs[i] + xs[i+1]) * 0.5
            my = (ys[i] + ys[i+1]) * 0.5
            cx = min(int(mx // cell_w), cols - 1)
            cy = min(int(my // cell_h), rows - 1)
            idx = cy * cols + cx
            seg_len = float(np.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i]))
            feats[idx, b] += seg_len

    # per-cell L1 normalize
    s = feats.sum(axis=1, keepdims=True)             # (R, 1)
    row_mask = (s[:, 0] > 0)                         # (R,)
    feats[row_mask] /= s[row_mask]                   # (k, C) / (k, 1)  -> OK

    return feats.reshape(-1).astype(np.float32)

def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (n + eps)

def build_wire_db_from_json(db_entries: List[dict], H: int, W: int, rows: int, cols: int,
                            score_th: float, num_bins: int,
                            keep_basenames: Optional[set] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return (db_mat (N,D), name2idx) aligned by basename."""
    feats, names = [], []
    for e in db_entries:
        fname = Path(_get_fname(e)).name
        if keep_basenames is not None and fname not in keep_basenames:
            continue
        lines = preprocess_lines_np(e, H, W, score_th)
        vec = compute_cell_features_np(lines, H, W, rows, cols, num_bins)
        if np.sum(vec) == 0:
            continue
        feats.append(vec)
        names.append(fname)
    if not feats:
        return np.zeros((0, rows*cols*num_bins), np.float32), {}
    mat = l2_normalize_rows(np.vstack(feats).astype(np.float32))
    name2idx = {n: i for i, n in enumerate(names)}
    return mat, name2idx

def build_wire_q_from_json(q_entries: List[dict], H: int, W: int, rows: int, cols: int,
                           score_th: float, num_bins: int,
                           keep_basenames: Optional[set] = None) -> Dict[str, np.ndarray]:
    """Return dict: basename -> (D,) L2-normalized."""
    out = {}
    for e in q_entries:
        fname = Path(_get_fname(e)).name
        if keep_basenames is not None and fname not in keep_basenames:
            continue
        lines = preprocess_lines_np(e, H, W, score_th)
        vec = compute_cell_features_np(lines, H, W, rows, cols, num_bins)
        if np.sum(vec) == 0:
            continue
        v = vec.astype(np.float32)
        n = np.linalg.norm(v) + 1e-6
        out[fname] = (v / n).astype(np.float32)
    return out

def chi2_distance_rowwise(q_vec: np.ndarray, db_mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    diff = db_mat - q_vec[None, :]
    denom = db_mat + q_vec[None, :] + eps
    chi2 = (diff * diff) / denom
    return chi2.sum(axis=1)

def chi2_to_similarity(d: np.ndarray, tau: float) -> np.ndarray:
    # map chi2 distance -> [0,1], higher is better
    tau = float(tau) if tau and tau > 0 else 1.0
    # 先用 float64 算 exp 再 cast，並裁剪指數避免下溢
    x = np.clip(d / tau, 0.0, 50.0)           # 50 對 float32 夠保守（exp(-50)≈1.9e-22）
    return np.exp(-x, dtype=np.float64).astype(np.float32)

def name2imgpath_from_graphs(names: List[str], graphs: List) -> Dict[str, str]:
    """
    嘗試從 graph metadata 取出影像路徑；若找不到則回傳 basename 自身（讓視覺化能至少印名稱）
    你的 JSON 若有不同欄位（如 'image', 'img_path'），這裡會盡量兼容。
    """
    out = {}
    for n, g in zip(names, graphs):
        p = None
        for key in ('image', 'img_path', 'path', 'file', 'rgb'):
            if key in g.graph:
                p = g.graph[key]
                break
        out[n] = p if p is not None else n
    return out

def load_graphs(json_path: Path):
    data = json.loads(json_path.read_text())
    names, graphs, descs_raw = [], [], []
    pad_to = None  # 可選參數：補零到固定長度；None 表示用本檔案內最大長度
    for entry in data:
        names.append(entry['image'])
        g = nx.Graph()
        # add nodes using 'center' field
        for node in entry['nodes']:
            cx, cy, cz = node['center']
            g.add_node(
                node['id'],
                category=node['category'],
                cx=float(cx), cy=float(cy), cz=float(cz),
                w=float(node['w']), h=float(node['h']),
                area=float(node['area']), depth=float(node['depth']))
        # add edges
        for edge in entry['edges']:
            g.add_edge(
                edge['from'], edge['to'],
                relation=edge['relation'],
                dist=float(edge['dist']), angle=float(edge['angle'])
            )

        graphs.append(g)
        d = entry.get("descriptor", None)
        descs_raw.append(_flatten_descriptor(d))

        # descs.append(entry['descriptor'])

    # 若未指定 pad_to，就用本檔案內的最大長度
    target_len = pad_to if pad_to is not None else max((v.size for v in descs_raw), default=0)

    # 統一補零到 target_len
    descs = np.stack([_pad1d(v, target_len) if v.size != target_len else v.astype(np.float32, copy=False)
                      for v in descs_raw], axis=0)

    return names, graphs, descs

def _flatten_descriptor(d):
    """把 descriptor 攤平成 1D float32 向量；支援 list/nested-list/dict。"""
    if d is None:
        return np.empty(0, dtype=np.float32)
    if isinstance(d, dict):
        parts = []
        # 為確保穩定順序，用 key 排序（若你有固定 key 順序，也可改成自訂順序）
        for k in sorted(d.keys()):
            parts.append(np.asarray(d[k], dtype=np.float32).ravel())
        return np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.float32)
    return np.asarray(d, dtype=np.float32).ravel()

def _pad1d(vec: np.ndarray, target_len: int):
    """不足補 0，超過不截斷（若要截斷可在這裡處理）。"""
    n = vec.size
    if n == target_len:
        return vec.astype(np.float32, copy=False)
    if n < target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[:n] = vec
        return out
    # 如果你想要硬截斷，改成 return vec[:target_len]
    return vec  # 保留原長度，後面 stack 前會統一長度

"""
---------------- Point Cloud IoU helpers (ported from eval_print_hw.py) ----------------
"""
def _resolve_img_path(item, root: Optional[Path] = None):
    """
    接受:
      - 字串（可能是完整路徑或相對於 root 的路徑/檔名）
      - 或 dict（會優先抓 'path'、'ref_image'、'name' 其中之一）
    回傳: 可交給 imread/open 的字串路徑（找不到也照樣回傳推導結果）
    """
    # 1) 取出字串
    if isinstance(item, dict):
        for k in ('path', 'ref_image', 'name'):
            if k in item and item[k]:
                item = item[k]
                break
        else:
            item = ""  # 沒東西就留空字串

    s = str(item)
    p = Path(s)

    # 2) 若本身就像是完整路徑，直接用
    if p.is_absolute() or p.exists():
        return s

    # 3) 否則嘗試用 root 拼接
    if root is not None:
        return str(Path(root) / s)

    # 4) 退而求其次，原字串照用（給上游 debug）
    return s

# ---------------- Rankers ----------------
def rank_seq_lcs_xy(q_seq_xy, c_seq_xy_list, top_k, wx=0.5, wy=0.5):
    """
    q_seq_xy: (seqxA, seqyA)
    c_seq_xy_list: list of (seqxB, seqyB)
    回傳：scores(相似度, 大到小), idxs(對應候選索引)
    """
    S_list = []
    for (sxB, syB) in c_seq_xy_list:
        S, Sx, Sy, D = seq_lcs_similarity_xy(q_seq_xy[0], sxB, q_seq_xy[1], syB, wx=wx, wy=wy)
        S_list.append(S)
    S_arr = np.asarray(S_list, dtype=np.float32)
    Nc = S_arr.shape[0]
    k_eff = min(top_k, Nc)
    # 由大到小（相似度越大越好）
    part = np.argpartition(-S_arr, k_eff - 1)[:k_eff]
    idx_sorted = part[np.argsort(-S_arr[part])]
    scores = S_arr[idx_sorted]
    return scores, idx_sorted

def rank_wireframe_only(q_name, wire_q_dict, wire_db_mat, wire_db_name2idx, cand_names, k, wire_tau):
    """
    只用 wireframe（χ²→similarity）打分。
    僅對能在 wire_db 中對得上 basename 的 candidates 計分。
    """
    from os.path import basename
    if (wire_q_dict is None) or (wire_db_mat is None) or (wire_db_mat.size == 0):
        return np.array([], dtype=np.float32), np.array([], dtype=int)

    # 查 query 的 wire 向量
    qv = wire_q_dict.get(Path(q_name).name)
    if qv is None:
        return np.array([], dtype=np.float32), np.array([], dtype=int)

    # 建立候選索引映射
    valid = []
    idx_map = []
    for i, cn in enumerate(cand_names):
        bn = Path(cn).name
        j = wire_db_name2idx.get(bn, None)
        if j is not None:
            valid.append(i)
            idx_map.append(j)
    if not valid:
        return np.array([], dtype=np.float32), np.array([], dtype=int)

    db_sub = wire_db_mat[np.array(idx_map, dtype=int), :]
    dists = chi2_distance_rowwise(qv, db_sub)
    # 自動 tau
    tau = np.median(dists) if (wire_tau is None or wire_tau <= 0) else wire_tau
    tau = float(tau) if tau > 1e-8 else 1.0
    sims = chi2_to_similarity(dists, tau)

    # 映回全庫索引
    valid = np.array(valid, dtype=int)
    part = np.argpartition(-sims, min(k, sims.size) - 1)[:min(k, sims.size)]
    order = np.argsort(-sims[part])
    idx_local = part[order]
    idx_global = valid[idx_local]
    return sims[idx_local], idx_global


def write_results_json(out_json: str, result_dict: Dict[str, List[dict]]):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

def write_results_csv(out_csv: str, result_dict: Dict[str, List[dict]]):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "rank", "candidate", "score", "image_path", "counts_q", "counts_c"])
        for qn, rows in result_dict.items():
            for r, row in enumerate(rows, 1):
                # 用 JSON 字串裝向量在 CSV 中
                cq = json.dumps(row.get("counts_q", []), ensure_ascii=False)
                cc = json.dumps(row.get("counts_c", []), ensure_ascii=False)
                w.writerow([qn, r, row.get("name",""), row.get("score",""), row.get("path",""), cq, cc])

#--------------Fusion Utils--------------
def _robust01(x, q_low=0.05, q_high=0.95):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    x_f = x[finite]
    q1 = np.quantile(x_f, q_low)
    q9 = np.quantile(x_f, q_high)
    x = np.clip(x, q1, q9)
    denom = max(q9 - q1, 1e-6)
    return (x - q1) / denom

def _ranks_desc(x):
    """相似度越大名次越前（1=最好）"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(-x)  # 大→小
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks

def _ranks_asc(x):
    """距離越小名次越前（1=最好）"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)   # 小→大
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks

def fuse_scores(seq_scores, wire_dists, args):
    """Score-level 融合：LCS相似度 + chi2→相似度→各自robust縮放→加權"""
    seq_scores = np.asarray(seq_scores, dtype=np.float32)   # S in [0,1]
    wire_dists = np.asarray(wire_dists, dtype=np.float32)   # chi2 distance

    # 決定 tau（0=auto: 用當前 query 的距離中位數；若無有效值則取1.0）
    finite = np.isfinite(wire_dists)
    if args.tau > 0:
        tau = float(args.tau)
    else:
        tau = float(np.median(wire_dists[finite])) if np.any(finite) else 1.0

    # 距離→相似度（使用你檔案中既有的轉換函式）
    wire_scores = chi2_to_similarity(wire_dists, tau)  # exp(-d/tau) ∈ (0,1]

    # 每個 query 單獨做 robust 縮放到 [0,1]
    s_seq  = _robust01(seq_scores,  args.q_low, args.q_high)
    s_wire = _robust01(wire_scores, args.q_low, args.q_high)

    # 權重 L1 正規化；若相加為 0 則回退到 0.5/0.5
    w_sum = args.w_seq + args.w_wire
    if w_sum <= 1e-12:
        w_seq = w_wire = 0.5
    else:
        w_seq  = args.w_seq  / w_sum
        w_wire = args.w_wire / w_sum

    # 若 wire 端無有效訊息，退場：只用 seq
    if not np.any(np.isfinite(wire_dists)):
        return s_seq

    return w_seq * s_seq + w_wire * s_wire

def rrf_fuse_ranks(seq_scores, wire_dists, c=60):
    """Rank-level 融合（RRF）：seq 用相似度降冪取名次；wire 用距離昇冪取名次"""
    seq_scores = np.asarray(seq_scores, dtype=np.float32)
    wire_dists = np.asarray(wire_dists, dtype=np.float32)

    r_seq  = _ranks_desc(seq_scores)
    # 若 wire 端無有效訊息，退場：只用 seq 排名
    if not np.any(np.isfinite(wire_dists)):
        return 1.0 / (c + r_seq.astype(np.float32))

    r_wire = _ranks_asc(wire_dists)

    return 1.0 / (c + r_seq.astype(np.float32)) + 1.0 / (c + r_wire.astype(np.float32))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_json', type=Path, required=True)
    parser.add_argument('--candidate_json', type=Path, required=True)
    parser.add_argument('--query_root', type=Path, required=True, help="absolute folder for query images")
    parser.add_argument('--candidate_root', type=Path, required=True, help="absolute folder for candidate images")
    parser.add_argument('--top_k', type=int, default=50)
# --- Wireframe options ---
    parser.add_argument('--wire_query_json', type=Path, default=None,
                        help="wireframe JSON for queries (same format as your wireframe code)")
    parser.add_argument('--wire_candidate_json', type=Path, default=None,
                        help="wireframe JSON for candidates (same format as your wireframe code)")
    parser.add_argument('--wire_target_hw', nargs=2, type=int, metavar=('H','W'), default=None,
                        help="target resize for wireframe feature (H W). If None, try to infer from data.")
    parser.add_argument('--wire_grid', nargs=2, type=int, metavar=('rows','cols'), default=[4, 4],
                        help="wireframe grid size")
    parser.add_argument('--wire_score_th', type=float, default=0.75,
                        help="score threshold for lines_score")
    parser.add_argument('--wire_bins', type=int, default=16,
                        help="orientation bins in [0,180)")
    parser.add_argument('--wire_tau', type=float, default=0.0,
                        help="chi2->sim decay; if <=0, auto-calibrate per query")

    parser.add_argument("--viz_flag", type=str, default=None,
                    choices=["node","edge","coarse","GED","Wireframe", "seq_wire"],
                    help="選擇要輸出的排名階段")
    parser.add_argument("--out_root", type=str, default="./viz",
                        help="視覺化與結果輸出的根資料夾")
    parser.add_argument("--class_order_override", type=str, default="",
                        help="以逗號分隔的 class 順序（留空則用預設順序）。")
    # ---- overlay 2D nodes text on tiles ----
    parser.add_argument("--mi_json", type=Path, default=None,
                    help="Mapping JSON: query → ref_image（只顯示比較，不參與排名）")
                    
    parser.add_argument("--coarse_output", type=Path, default=None,
                    help="MI-style JSON output: {'results': {query_img: [{'ref_image': cand_img}, ...]}}")

    # --- Fusion options ---
    parser.add_argument('--fusion', choices=['score', 'rrf'], default='score',
                        help='Fusion strategy when viz_flag=seq_wire')
    parser.add_argument('--w_seq', type=float, default=0.5, help='Weight for Node Sequence (score-level)')
    parser.add_argument('--w_wire', type=float, default=0.5, help='Weight for Wireframe (score-level)')
    parser.add_argument('--tau', type=float, default=0.0, 
                        help='Chi-square temperature; 0 means auto per-query (median distance)')
    parser.add_argument('--q_low', type=float, default=0.05, help='Lower quantile for robust scaling')
    parser.add_argument('--q_high', type=float, default=0.95, help='Upper quantile for robust scaling')
    parser.add_argument('--rrf_c', type=int, default=60, help='RRF constant c for rank-level fusion')

    args = parser.parse_args()

    # Construct real graph
    real_main()

    # Load graphs
    q_names, q_graphs, q_descs = load_graphs(args.query_json)
    c_names, c_graphs, c_descs = load_graphs(args.candidate_json)

    # --- 在進入視覺化之前，印出候選集數量 ---
    print(f"[i] Number of candidates after filtering: {len(c_names)}")
    cand_base2idx = { _canon_name(name): i for i, name in enumerate(c_names) }
    for i, name in enumerate(c_names):
        cand_base2idx[_canon_name(Path(name).name)] = i

    # 依 override 或預設建立顯示順序
    if args.class_order_override.strip():
        class_order = [s.strip() for s in args.class_order_override.split(",") if s.strip()]
    else:
        class_order = list(STRUCTURAL_CLASSES)

    # 名稱→索引（全名與 basename 雙保險）
    name2idx_for_counts = {name: i for i, name in enumerate(c_names)}
    for i, name in enumerate(c_names):
        name2idx_for_counts[Path(name).name] = i

    node_len = 64

    # 供視覺化顯示（node 向量值）使用
    cand_name2idx = {name: i for i, name in enumerate(c_names)}
    for i, name in enumerate(c_names):
        cand_name2idx[Path(name).name] = i

    q_paths = [str(args.query_root / n) for n in q_names]
    c_paths = [str(args.candidate_root / n) for n in c_names]

    

    # ===== 可視化五模式：若指定 viz_flag 就執行並 return =====
    if args.viz_flag is not None:
        flag = args.viz_flag
        out_dir = Path(args.out_root) / flag.lower()
        out_collage = out_dir / "collages"
        out_json = out_dir / f"results_{flag.lower()}.json"
        out_csv  = out_dir / f"results_{flag.lower()}.csv"
        os.makedirs(out_collage, exist_ok=True)

        # 準備影像路徑（視覺化用）
        q_name2img = name2imgpath_from_graphs(q_names, q_graphs)
        c_name2img = name2imgpath_from_graphs(c_names, c_graphs)

        # 若使用 wireframe 模式，確保 wire 特徵已建好（沿用你原本的 build 函式與參數）
        wire_db_mat = None
        wire_db_name2idx = None
        wire_q_dict = None
        if flag == "Wireframe" or flag == "seq_wire":
            # 需預先以你的參數構建 wire 資料庫與 query 字典
            # 假設你原本參數名：args.wire_db_json, args.wire_q_json, args.wire_H/W/rows/cols/score_th/num_bins
            if args.wire_candidate_json and args.wire_query_json:
                db_entries = load_wire_json(Path(args.wire_candidate_json))
                q_entries  = load_wire_json(Path(args.wire_query_json))

                # pick target size
                if args.wire_target_hw is not None:
                    H, W = map(int, args.wire_target_hw)
                else:
                    # try to infer from first query entry, fallback to (480,640)
                    if len(q_entries) > 0:
                        H = int(q_entries[0].get("height", 1080))
                        W = int(q_entries[0].get("width",  1920))
                    else:
                        H, W = 1080, 1920

                rows, cols   = map(int, args.wire_grid)
                score_th     = float(args.wire_score_th)
                num_bins     = int(args.wire_bins)

                # filter to current SG names (by basename)
                q_basenames = {Path(p).name for p in q_paths}
                c_basenames = {Path(p).name for p in c_paths}


                wire_db_mat, wire_db_name2idx = build_wire_db_from_json(
                    db_entries, H, W, rows, cols,
                    score_th, num_bins, keep_basenames=c_basenames
                )
                wire_q_dict = build_wire_q_from_json(
                    q_entries, H, W, rows, cols,
                    score_th, num_bins, keep_basenames=q_basenames
                )

        # 依旗標執行
        TOPK = 50
        results = {}
        mi_map = load_mi_map(args.mi_json) if args.mi_json else {}
        args._ref_map = mi_map

        for qi, qn in enumerate(q_names):
            q_desc = q_descs[qi]
            q_img  = q_name2img.get(qn, qn)
            q_node_vec = (q_node_mat[qi] if q_node_mat is not None else None)

            if flag == "seq_wire":
                # --- 1) 先備妥 Node Sequence 的 2D 投影向量（一次建好快取，避免重算） ---
                # if 'Q_SEQ_XY_CACHE' not in globals():
                #     global Q_SEQ_XY_CACHE, C_SEQ_XY_CACHE
                Q_SEQ_XY_CACHE = build_seq_xy_batch(q_graphs, class_order, intrinsics="real")
                C_SEQ_XY_CACHE = build_seq_xy_batch(c_graphs, class_order, intrinsics="syn")

                q_seq_xy = Q_SEQ_XY_CACHE[qi]
                Nc = len(C_SEQ_XY_CACHE)


                # 以「取滿庫」的方式拿到 Node Sequence 的分數（相似度越大越好）
                seq_scores_full, idxs_full = rank_seq_lcs_xy(
                    q_seq_xy, C_SEQ_XY_CACHE, top_k=Nc, wx=0.5, wy=0.5
                )
                seq_scores = np.full(Nc, -np.inf, dtype=np.float32)
                seq_scores[idxs_full] = seq_scores_full  # 對齊到 candidates 全庫順序

                # --- 2) 準備 Wireframe 的 χ²距離向量（同樣對齊到 candidates 全庫順序） ---
                wire_dists = np.full(Nc, np.inf, dtype=np.float32)
                if (wire_q_dict is not None) and (wire_db_mat is not None) and (wire_db_mat.size > 0):
                    qb = Path(q_names[qi]).name
                    qv = wire_q_dict.get(qb, None)
                    if qv is not None:
                        # 依 candidates 名稱找出對應到 wire_db 的列索引
                        wire_idx_of_c = np.array([wire_db_name2idx.get(Path(n).name, -1) for n in c_names], dtype=int)
                        m = wire_idx_of_c >= 0
                        if np.any(m):
                            d_all = chi2_distance_rowwise(qv, wire_db_mat[wire_idx_of_c[m], :])  # 僅算對得上的
                            wire_dists[m] = d_all.astype(np.float32)

                # --- 3) 融合：Score-level 或 Rank-level（沿用你已定義的函式與 argparse 參數） ---
                if getattr(args, "fusion", "score") == "rrf":
                    fused = rrf_fuse_ranks(seq_scores, wire_dists, c=int(getattr(args, "rrf_c", 60)))
                else:
                    fused = fuse_scores(seq_scores, wire_dists, args)

                k_eff = min(TOPK, Nc)
                part = np.argpartition(-fused, k_eff - 1)[:k_eff]
                order = np.argsort(-fused[part])
                idxs = part[order]
                scores = fused[idxs].astype(np.float32)

                top1 = int(idxs[0])

                print("\n========== [DEBUG layout] ==========")
                print(f"Query image: {q_names[qi]}")
                print(f"Top-1 cand : {c_names[top1]}")
            else:
                raise ValueError(f"Unknown viz_flag: {flag}")
            
            # 名稱→索引
            cand_name2idx = {name:i for i,name in enumerate(c_names)}
            for i,name in enumerate(c_names):
                cand_name2idx[Path(name).name] = i

            # 整理輸出 rows
            rows = []
            scores_map = {int(i): float(s) for i, s in zip(idxs, scores)}

            for r, ci in enumerate(idxs, 1):
                cn = c_names[ci]
                row = {
                    "rank": r,
                    "name": cn,
                    "score": float(scores[r-1]) if r-1 < len(scores) else None,
                    "path": c_name2img.get(cn, cn)
                }

                rows.append(row)
            results[qn] = rows


            # cand name → idx
            cand_name2idx = {name: i for i, name in enumerate(c_names)}
            for i, name in enumerate(c_names):
                cand_name2idx[Path(name).name] = i

            # class 名稱與統計維度
            class_names = list(STRUCTURAL_CLASSES)
            stat_per_class = STAT_PER_CLASS

            # 建 cand 名稱 → graph（全名與 basename 都建）
            cand_name2graph = {name: g for name, g in zip(c_names, c_graphs)}
            for name, g in zip(c_names, c_graphs):
                cand_name2graph[Path(name).name] = g

            # 優先用完整 query 路徑（避免 missing）
            q_img = q_name2img.get(q_names[qi], q_names[qi])

            #---
            # [C-1] 取得這張 query 的 ref 名稱
            q_canon = _canon_name(q_names[qi])
            ref_canon = mi_map.get(q_canon, None)

            # === 產生 ref_item（不重算，直接用當前候選 rows 的現成分數；不在 top-K 就顯示圖 + N/A） ===
            ref_item = None
            if ref_canon is not None:
                # 1) 先在本張 query 的 top-K 候選 rows 裡找看看
                matched_row = None
                for row in rows:
                    # rows[i]["name"] 可能是完整路徑或相對路徑，統一轉成 canon 後比對
                    try:
                        nm_canon = _canon_name(row.get("name", ""))
                    except Exception:
                        nm_canon = ""
                    if nm_canon == ref_canon:
                        matched_row = row
                        break

                if matched_row is not None:
                    ref_item = {
                        "name": matched_row.get("name", "ref"),
                        "path": matched_row.get("path", matched_row.get("name")),
                        "score": matched_row.get("score"),   # 現成分數
                        "pc_iou": matched_row.get("pc_iou")
                    }
                else:
                    # 優先用 c_name2img 對應出可讀取的影像路徑；沒有就用 candidate_root/ref_canon.*
                    ref_img_path = c_name2img.get(ref_canon)
                    if (not ref_img_path) and args.candidate_root:
                        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                            p = Path(args.candidate_root) / (ref_canon + ext)
                            if p.exists():
                                ref_img_path = str(p)
                                break

                    ref_item = {
                        "name": ref_canon,
                        "path": ref_img_path if ref_img_path else ref_canon,
                        "score": None,
                        "pc_iou": None
                    }
            #---

        # 額外輸出 MI-style JSON：query_image → [{"ref_image": candidate_image}, ...]
        if args.coarse_output:
            mi_results = {"results": {}}
            for qn in q_names:
                rows = results.get(qn, [])
                # 將 query 名稱解析為完整影像路徑（用 query_root 變成絕對路徑）
                q_img_path = _resolve_img_path(qn, args.query_root)
                ref_list = []
                for row in rows:
                    # 用 candidate_root 把 candidate 解析成完整路徑
                    ref_path = _resolve_img_path(row, args.candidate_root)
                    ref_list.append({"ref_image": ref_path})
                mi_results["results"][q_img_path] = ref_list

            out_coarse = str(args.coarse_output)
            os.makedirs(os.path.dirname(out_coarse), exist_ok=True)
            with open(out_coarse, "w", encoding="utf-8") as f:
                json.dump(mi_results, f, ensure_ascii=False, indent=2)
            print(f"[{flag}] MI-style results saved to: {out_coarse}")


        # 寫 JSON 與 CSV
        write_results_json(str(out_json), results)
        write_results_csv(str(out_csv), results)

        print(f"[{flag}] results saved to: {out_json} and {out_csv}")
        print(f"[{flag}] collages saved to: {out_collage}")
        return

if __name__ == '__main__':
    main()
