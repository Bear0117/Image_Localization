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

# Test
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
DELTA = 1.0
EPSILON = 1.0
LAMBDA_NODE = 1.0
LAMBDA_EDGE = 0.5
# n = 1.0
# e = 1.0
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

def _class_to_id_map(class_order):
    return {c: i for i, c in enumerate(class_order)}

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

    # 1) 內參選擇（與 _build_layout_counts_one 的 auto 規則一致）
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

def _build_layout_counts_one(
    graph,
    class_order: List[str],
    rows: int = LAYOUT_ROWS,
    cols: int = LAYOUT_COLS,
    intrinsics: str = "real",   # "real" | "syn" | "auto"
) -> np.ndarray:
    """
    以 node 的 3D (cx,cy,cz) → 2D (u,v) 投影到 rows×cols 的格子，累計各類別 count。
    回傳 shape (rows*cols*|class_order|,) 的 float32 向量（整數值）。
    """
    # 1) 內參選擇
    if intrinsics == "syn":
        fx, fy, cx, cy = FX_SYN, FY_SYN, CX_SYN, CY_SYN
    elif intrinsics == "real":
        fx, fy, cx, cy = FX_REAL, FY_REAL, CX_REAL, CY_REAL
    else:  # "auto"：依檔名/來源猜測（若 graph.graph 有 image / path 可用）
        img_name = str(graph.graph.get("image", graph.graph.get("img_path", ""))).lower()
        is_syn = ("_rgb_image" in img_name) or img_name.endswith(".png")
        fx, fy, cx, cy = (FX_SYN, FY_SYN, CX_SYN, CY_SYN) if is_syn else (FX_REAL, FY_REAL, CX_REAL, CY_REAL)

    # 2) 準備
    C = len(class_order)
    cls2i = {c: i for i, c in enumerate(class_order)}
    vec = np.zeros((rows * cols * C,), dtype=np.float32)

    # 3) 蒐集 X,Y 範圍（給 fallback 正規化用）
    xs, ys = [], []
    for _, data in graph.nodes(data=True):
        if ("cx" in data) and ("cy" in data):
            xs.append(float(data["cx"]))
            ys.append(float(data["cy"]))
    min_x = min(xs) if xs else 0.0
    max_x = max(xs) if xs else 1.0
    min_y = min(ys) if ys else 0.0
    max_y = max(ys) if ys else 1.0
    rx = max(max_x - min_x, 1e-6)
    ry = max(max_y - min_y, 1e-6)

    cell_w = IMG_W / cols
    cell_h = IMG_H / rows

    # 4) 逐點投影/累計
    for _, data in graph.nodes(data=True):
        cat = data.get("category")
        if cat not in cls2i:
            continue
        if not all(k in data for k in ("cx", "cy", "cz")):
            continue

        X, Y, Z = float(data["cx"]), float(data["cy"]), float(data["cz"])
        uv = _project_uv((X, Y, Z), fx, fy, cx, cy)

        if uv is None:
            # Z 無效：退回以 (cx,cy) 做 min–max 到畫布
            u = ( (X - min_x) / rx ) * IMG_W
            v = ( (Y - min_y) / ry ) * IMG_H
        else:
            u, v = uv

        gx = int(u // cell_w)
        gy = int(v // cell_h)
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            # 超界：丟棄（或改成夾緊）
            continue

        off = (gy * cols + gx) * C + cls2i[cat]
        vec[off] += 1.0

    return vec

# [A-2] 小工具：canonical 命名 + 讀 MI
def _canon_name(s: str) -> str:
    b = Path(s).name.lower()
    if "." in b:
        b = b.rsplit(".", 1)[0]
    if b.endswith("_rgb_image"):
        b = b[:-len("_rgb_image")]
    return b

def load_mi_map(mi_path: Path) -> dict[str, str]:
    if not mi_path or not mi_path.exists():
        return {}
    data = json.loads(mi_path.read_text(encoding="utf-8"))
    out = {}
    for q, arr in data.get("results", {}).items():
        if isinstance(arr, list) and arr and isinstance(arr[0], dict):
            r = arr[0].get("ref_image")
            if r:
                out[_canon_name(q)] = _canon_name(r)
    return out

def _measure_multiline(draw: ImageDraw.Draw, text: str, font, line_spacing: int = 2) -> Tuple[int, int]:
    """相容舊 Pillow：量測多行文字大小。"""
    try:
        bbox = draw.multiline_textbbox((0,0), text, font=font, spacing=line_spacing, align="left")
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        # fallback：逐行 textsize 疊加
        w = h = 0
        for i, line in enumerate(text.split("\n")):
            w_i, h_i = draw.textsize(line, font=font)
            w = max(w, w_i)
            h += h_i + (line_spacing if i > 0 else 0)
        return w, h

def build_query_ref_header(
    q_img_path: str,
    q_name: str,
    ref_img_path: Optional[str],
    ref_name: str,
    ref_score: Optional[float],
    ref_iou: Optional[float],
    *,
    cell_h: int,
    pad: int,
    font,
    font_b,
    text_panel_min_w: int = 220,
    q_im_override: 'Optional[Image.Image]' = None,
) -> Tuple[Image.Image, int, int]:
    """
    產生「不覆蓋圖片」的上方標頭條（左→右：query圖、query檔名區、ref圖、ref細節區）
    回傳：(header_img, header_width, header_height)
    """
    if ref_img_path:
        try:
            r_im = _safe_open_image(ref_img_path, target_h=cell_h)
        except Exception:
            r_im = Image.new("RGB", (cell_h, cell_h), (220,220,220))
    else:
        r_im = Image.new("RGB", (cell_h, cell_h), (220,220,220))
        
    if (q_im_override is not None) and isinstance(q_im_override, Image.Image):
        # 若外面已經把節點畫到 q_im 上，就直接用它；大小不對時等比縮到 cell_h
        if q_im_override.size[1] != cell_h:
            ratio = cell_h / float(q_im_override.size[1])
            new_w = max(1, int(round(q_im_override.size[0] * ratio)))
            q_im = q_im_override.resize((new_w, cell_h), Image.BILINEAR)
            # print(f"Debug: resize q_im_override from {q_im_override.size} to {q_im.size}")
        else:
            q_im = q_im_override
            # print(f"Debug: use q_im_override size={q_im.size}")
    else:
        q_im = _safe_open_image(q_img_path, target_h=cell_h)

    
        
    # # 1) 開圖（高度等於 cell_h；寬度稍後計算）
    # q_im = _safe_open_image(q_img_path, target_h=cell_h)
    # if ref_img_path:
    #     try:
    #         r_im = _safe_open_image(ref_img_path, target_h=cell_h)
    #     except Exception:
    #         r_im = Image.new("RGB", (cell_h, cell_h), (220,220,220))
    # else:
    #     r_im = Image.new("RGB", (cell_h, cell_h), (220,220,220))

    # 2) 預備兩個純文字面板：query 檔名、ref 細節
    q_name_disp = (q_name if len(q_name) <= 50 else (q_name[:47] + "…"))
    ref_name_disp = (ref_name if len(ref_name) <= 50 else (ref_name[:47] + "…"))
    line1 = "[REF] " + ref_name_disp
    line2 = "score={:.4f}".format(ref_score) if isinstance(ref_score, (int,float)) else "score=NA"
    line3 = "IoU(pc)={:.3f}".format(ref_iou) if isinstance(ref_iou, (int,float)) else None
    ref_text = line1 + "\n" + line2 + (("\n" + line3) if line3 else "")

    # 3) 計算面板寬度
    # 先建立暫時畫布來量文字
    tmp = Image.new("RGB", (10,10), (255,255,255))
    tdraw = ImageDraw.Draw(tmp)
    q_text_w, q_text_h = _measure_multiline(tdraw, q_name_disp, font=font_b)
    ref_text_w, ref_text_h = _measure_multiline(tdraw, ref_text, font=font)

    q_panel_w   = max(text_panel_min_w, q_text_w + pad*2)
    ref_panel_w = max(text_panel_min_w, ref_text_w + pad*2)

    header_w = pad + q_im.size[0] + pad + q_panel_w + pad + r_im.size[0] + pad + ref_panel_w + pad
    header_h = cell_h

    # 4) 繪製 header
    header = Image.new("RGB", (header_w, header_h), (255,255,255))
    draw = ImageDraw.Draw(header)

    # 位置：左起 query 圖
    x = pad
    header.paste(q_im, (x, (header_h - q_im.size[1]) // 2))
    x += q_im.size[0] + pad

    # query 名稱面板（白底黑字）
    q_tx = x + pad
    q_ty = (header_h - q_text_h) // 2
    draw.text((q_tx, q_ty), q_name_disp, font=font_b, fill=(0,0,0))
    x += q_panel_w + pad

    # ref 圖
    header.paste(r_im, (x, (header_h - r_im.size[1]) // 2))
    x += r_im.size[0] + pad

    # ref 細節面板
    ref_tx = x + pad
    # 多行排版
    cur_y = (header_h - ref_text_h) // 2
    draw.text((ref_tx, cur_y), "[REF] " + ref_name_disp, font=font_b, fill=(0,0,0)); 
    cur_y += font_b.size + 4
    draw.text((ref_tx, cur_y), line2, font=font, fill=(20,20,20)); 
    cur_y += font.size + 4
    if line3:
        draw.text((ref_tx, cur_y), line3, font=font, fill=(20,20,20))

    return header, header_w, header_h
# ---------------- Layout helpers (batch) ----------------

def build_layout_counts_batch(graphs, class_order: List[str], rows: int = LAYOUT_ROWS, cols: int = LAYOUT_COLS , intrinsics: str = "real") -> np.ndarray:
    mats = [ _build_layout_counts_one(g, class_order, rows, cols, intrinsics=intrinsics) for g in graphs ]
    return np.stack(mats, axis=0).astype(np.float32)

def chi2_distance_all(q_vec: np.ndarray, C_mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q_vec, dtype=np.float32).reshape(1,-1)
    C = np.asarray(C_mat, dtype=np.float32)
    diff = C - q
    denom = C + q + eps
    d = 0.5 * np.sum((diff * diff) / denom, axis=1, dtype=np.float64)
    return d.astype(np.float32)

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

# def chi2_to_similarity(d: np.ndarray, tau: float) -> np.ndarray:
#     """map chi2 distance -> [0,1], higher is better"""
#     return np.exp(-float(tau) * np.clip(d, 0.0, None)).astype(np.float32)

def chi2_to_similarity(d: np.ndarray, tau: float) -> np.ndarray:
    # map chi2 distance -> [0,1], higher is better
    tau = float(tau) if tau and tau > 0 else 1.0
    # 先用 float64 算 exp 再 cast，並裁剪指數避免下溢
    x = np.clip(d / tau, 0.0, 50.0)           # 50 對 float32 夠保守（exp(-50)≈1.9e-22）
    return np.exp(-x, dtype=np.float64).astype(np.float32)

def node_cost(attr1: dict, attr2: dict) -> float:
    # attr1 and attr2 are node attribute dicts from networkx
    type_cost = 0.0 if attr1.get('category') == attr2.get('category') else 1.0
    a1 = np.array([attr1.get('depth',0)], dtype=float)
    a2 = np.array([attr2.get('depth',0)], dtype=float)
    attr_cost = float(np.sum(np.abs(a1 - a2)))
    pos_cost = math.hypot(attr1.get('cx',0)-attr2.get('cx',0), attr1.get('cy',0)-attr2.get('cy',0), attr1.get('cz',0)-attr2.get('cz',0))
    total_cost = ALPHA * type_cost + BETA * attr_cost + GAMMA * pos_cost
    return total_cost

# Edge substitution cost
def edge_cost(attr1: dict, attr2: dict) -> float:
    # attr1 and attr2 are edge attribute float
    rel_cost = 0.0 if attr1.get('relation') == attr2.get('relation') else 1.0
    dist_cost = abs(attr1.get('dist',0) - attr2.get('dist',0))
    angle_diff = abs(attr1.get('angle',0) - attr2.get('angle',0))
    angle_cost = min(angle_diff, 360.0-angle_diff) / 180.0
    geom_cost = dist_cost + angle_cost
    total_cost = DELTA * rel_cost + EPSILON * geom_cost
    return total_cost

def ged_cost(qg: nx.Graph, cg: nx.Graph, timeout: float=1.0) -> float:
    e_iter = nx.optimize_graph_edit_distance(
        qg, cg,
        node_subst_cost=node_cost,
        node_del_cost=lambda n: LAMBDA_NODE,
        # node_ins_cost=lambda n: LAMBDA_NODE,
        node_ins_cost=lambda n: 0.0,
        edge_subst_cost=edge_cost,
        edge_del_cost=lambda e: LAMBDA_EDGE,
        # edge_ins_cost=lambda e: LAMBDA_EDGE,
        edge_ins_cost=lambda e: 0.0,
        edge_match=lambda e1, e2: e1.get('relation')==e2.get('relation')
    )
    try:
        return next(e_iter)
    except ValueError as e:
        return float('inf')
    except StopIteration:
        return float('inf')

def split_node_edge_desc(desc_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    依 STRUCTURAL_CLASSES 與 STAT_PER_CLASS 切出 node / edge 子向量矩陣
    回傳: (node_mat, edge_mat, node_len)
    """
    C = len(STRUCTURAL_CLASSES)
    node_len = C * (1 + STAT_PER_CLASS)  # 類別直方圖 + 每類統計
    node_mat = desc_mat[:, :node_len]
    edge_mat = desc_mat[:, node_len:]
    return node_mat, edge_mat, node_len

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

def cosine_topk_choose(q_vec, cand_mat, k, use_mask=False):
    if use_mask:
        return cosine_topk_masked(q_vec, cand_mat, k)
    else:
        return cosine_topk_full(q_vec, cand_mat, k)

def cosine_topk_full(q_vec, cand_mat, k, eps=1e-8):
    """
    q_vec    : (D,)   Query 1-D array（完整向量，不做遮罩）
    cand_mat : (Nc,D) Candidate matrix
    k        : top-k

    回傳:
      sims_top : (1,k)  cosine 相似度, 介於 [-1,1]
      idx_top  : (1,k)  對應的 candidate index
    """
    # 邊界處理
    if cand_mat.size == 0 or k <= 0:
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=int)

    Nc = cand_mat.shape[0]
    k_eff = min(k, Nc)

    # 轉成期望型別/形狀
    q = np.asarray(q_vec, dtype=np.float32).reshape(-1)
    C = np.asarray(cand_mat, dtype=np.float32)

    # 若 query 為全零或幾乎無能量，回傳 -1 分數（不中斷主流程）
    q_norm = np.linalg.norm(q)
    if q_norm < eps:
        idx = np.arange(k_eff, dtype=int)
        sims = np.full(k_eff, -1.0, dtype=np.float32)
        return sims[None, :], idx[None, :]

    # 全向量餘弦相似度
    C_norms = np.linalg.norm(C, axis=1) + eps  # 保護除零
    sims = (C @ q) / (C_norms * (q_norm + eps))  # (Nc,)

    # 用 argpartition 取 top-k（O(N)）再在子集合排序
    part = np.argpartition(-sims, k_eff - 1)[:k_eff]
    order = np.argsort(-sims[part])
    idx_sorted = part[order]

    return sims[idx_sorted][None, :].astype(np.float32), idx_sorted[None, :]

def cosine_topk_masked(q_vec, cand_mat, k):
    """
    q_vec      : (D,)   Query 1-D array (已移除固定維度)
    cand_mat   : (Nc,D) Candidate matrix
    k          : top‑k
    
    回傳:
      sims_top : (1,k)  cosine 相似度, 介於 [-1,1]
      idx_top  : (1,k)  對應的 candidate index
    """

    Nc = cand_mat.shape[0]
    if Nc == 0:
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=int)

    k_eff = max(1, min(k, Nc))

    mask = q_vec != 0                        # 動態遮罩

    if mask.sum() == 0:                      # 全為 0 → 無資訊
        idx_top = np.arange(k_eff, dtype=int)
        sims_top = np.full(k_eff, -1.0, dtype=np.float32)
        return sims_top[None, :], idx_top[None, :]

    q_sub   = q_vec[mask]                    # (d',)
    c_sub   = cand_mat[:, mask]              # (Nc,d')

    q_norm  = np.linalg.norm(q_sub) + 1e-8
    c_norms = np.linalg.norm(c_sub, axis=1) + 1e-8
    sims    = (c_sub @ q_sub) / (c_norms * q_norm)  # (Nc,)

    # idx_top_unsorted = np.argpartition(-sims, k)[:k]
    # idx_sorted = idx_top_unsorted[np.argsort(-sims[idx_top_unsorted])]

    part = np.argpartition(-sims, k_eff - 1)[:k_eff]
    idx_sorted = part[np.argsort(-sims[part])]   

    return sims[idx_sorted][None, :].astype(np.float32), idx_sorted[None, :]


"""
---------------- Point Cloud IoU helpers (ported from eval_print_hw.py) ----------------
"""
def compute_pc_iou_metrics(results: Dict[str, List[dict]],
                           iou_th: float = 0.5) -> Tuple[dict, List[dict]]:
    """
    results: { query_name: [ { 'rank':int, 'pc_iou':float or None, ... }, ... ] }
    回傳：
      summary: 總表
      details: 每個 query 的細表（可輸出 CSV）
    """
    details = []
    for qname, rows in results.items():
        # 以 rank 排好，確保 "第一張" 是 rank 最小者
        srows = sorted(rows, key=lambda r: r.get("rank", 1_000_000))

        count_ge = 0
        first_rank = None
        for r in srows:
            iou = r.get("pc_iou")
            if iou is not None and iou >= iou_th:
                count_ge += 1
                if first_rank is None:
                    first_rank = r.get("rank")

        hit = first_rank is not None
        recip = (1.0 / first_rank) if (first_rank is not None and first_rank > 0) else 0.0
        details.append({
            "query": qname,
            "hit": bool(hit),
            "count_ge_th": int(count_ge),
            "first_rank": int(first_rank) if first_rank is not None else None,
            "reciprocal_rank": float(recip),
        })

    num_q = len(details)
    hit_q = sum(1 for d in details if d["hit"])
    avg_cnt = (sum(d["count_ge_th"] for d in details) / num_q) if num_q else 0.0
    mrr = (sum(d["reciprocal_rank"] for d in details) / num_q) if num_q else 0.0
    hit_rate = (hit_q / num_q) if num_q else 0.0

    summary = {
        "num_queries": num_q,
        "iou_threshold": float(iou_th),
        "queries_with_at_least_one_hit": int(hit_q),   # (1) 多少張 query 至少有一張 IoU>=th
        "hit_rate": float(hit_rate),                   # 比例
        "avg_retrieved_per_query": float(avg_cnt),     # (2) 平均每張有幾張 IoU>=th
        "mrr": float(mrr),                             # (3) MRR（第一張 IoU>=th 的 rank 倒數之平均）
    }
    return summary, details

def load_point_cloud(filename: str) -> Optional[np.ndarray]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.npy':
        try:
            data = np.load(filename, allow_pickle=True).item()
        except Exception as e:
            raise RuntimeError(f"Failed to load point cloud from {filename}: {e}")
        if isinstance(data, dict) and "data" in data:
            point_cloud = data["data"]
            return np.array(point_cloud)
        else:
            return None
    elif ext == '.ply':
        pcd = o3d.io.read_point_cloud(filename)
        return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def compute_point_cloud_overlap(query_points: np.ndarray,
                                candidate_points: np.ndarray,
                                distance_threshold: float) -> Optional[float]:
    try:
        query_tree = cKDTree(query_points)
        candidate_tree = cKDTree(candidate_points)
    except Exception:
        return None

    # query -> candidate
    distances_q2c, _ = candidate_tree.query(query_points, distance_upper_bound=distance_threshold)
    matched_query_indices = set(np.where(distances_q2c != np.inf)[0])

    # candidate -> query
    distances_c2q, _ = query_tree.query(candidate_points, distance_upper_bound=distance_threshold)
    matched_candidate_indices = set(np.where(distances_c2q != np.inf)[0])

    # Note: this is a union of matched indices across both directions
    intersection = len(matched_query_indices.union(matched_candidate_indices))
    union = len(query_points) + len(candidate_points) - intersection
    if union == 0:
        return 0.0
    return intersection / union

def pc_iou_voxel(query_points, candidate_points, voxel_size: float=0.05) -> float:
    if len(query_points)==0 or len(candidate_points)==0:
        return 0.0
    mins = np.minimum(query_points.min(0), candidate_points.min(0))
    keys1 = np.floor((query_points - mins)/voxel_size).astype(np.int64)
    keys2 = np.floor((candidate_points - mins)/voxel_size).astype(np.int64)
    set1, set2 = set(map(tuple, keys1)), set(map(tuple, keys2))
    if not set1 and not set2: return 0.0
    inter = len(set1 & set2); uni = len(set1 | set2)

    return inter/uni

def compute_pc_iou_for_pair(query_image_path: str,
                            candidate_image_path: str,
                            voxel_size: float = 10.0) -> Optional[float]:
    q_stem = Path(query_image_path).stem
    c_stem = Path(candidate_image_path).stem
    if c_stem.endswith("_rgb_image"):
        c_stem = c_stem.replace("_rgb_image", "")

    # Path for hallway area
    # query_pc_path = f"/D/lulu/home/Delta/graph/New_GT_PC/Hallway/_{q_stem}_pointcloud.npy"
    # cand_pc_path  = f"/D/hoa/Delta_project/Dataset_0615/6F_pointcloud/{c_stem}_pointcloud.npy"

    # Path for elevator area
    # query_pc_path = f"/D/lulu/home/Delta/graph/New_GT_PC/Elevator/_{q_stem}_pointcloud.npy"
    # cand_pc_path  = f"/D/hoa/Delta_project/Dataset_0615/6F_pointcloud/{c_stem}_pointcloud.npy"

    # Path for r611 area
    query_pc_path = f"/D/lulu/home/Delta/graph/New_GT_PC/R611/_{q_stem}_pointcloud.npy"
    cand_pc_path  = f"/D/hoa/Delta_project/Dataset_0615/6F_pointcloud/{c_stem}_pointcloud.npy"  

    if not (os.path.exists(query_pc_path) and os.path.exists(cand_pc_path)):
        return None

    q_points = load_point_cloud(query_pc_path)
    c_points = load_point_cloud(cand_pc_path)
    if q_points is None or c_points is None:
        return None
    if q_points.size == 0 or c_points.size == 0:
        return None

    # return compute_point_cloud_overlap(q_points, c_points, distance_threshold)
    # print(voxel_size)
    return pc_iou_voxel(q_points, c_points, voxel_size)

def _safe_open_image(path: str, target_h=180):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        # 用灰底佔位圖
        im = Image.new("RGB", (target_h*4//3, target_h), (200, 200, 200))
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), "missing", fill=(50, 50, 50))
    # 等高縮放
    w, h = im.size
    if h != target_h:
        new_w = int(w * (target_h / h))
        im = im.resize((new_w, target_h), Image.BILINEAR)
    return im

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < eps or bn < eps:
        return float("nan")
    return float(np.dot(a, b) / ((an + eps) * (bn + eps)))

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

def visualize_grid_page(query_item,
                        top_ref_items,
                        out_png: str,
                        rank_offset: int,
                        query_root: Optional[Path] = None,
                        cand_root: Optional[Path] = None,
                        cols: int = 10,
                        rows: int = 1,
                        pad: int = 8,
                        # 既有 node 可視化用參數（可保留不動）
                        q_node_vec: Optional[np.ndarray] = None,
                        cand_descs: Optional[np.ndarray] = None,
                        cand_name2idx: Optional[dict] = None,
                        node_len: Optional[int] = None,
                        viz_node_mode: str = "both",
                        # 版面與標籤（已有的保留）
                        cell_w_override: Optional[int] = None,
                        label_classes: bool = False,
                        label_max_chars: int = 8,
                        stat_per_class: int = 5,
                        class_names: Optional[list] = None,
                        q_layout_vec: Optional[np.ndarray] = None,
                        c_layout_mat: Optional[np.ndarray] = None,
                        layout_rows: int = LAYOUT_ROWS,
                        layout_cols: int = LAYOUT_COLS,
                        layout_classes: Optional[List[str]] = None,
                        # ★ 新增：顯示 counts（未正規化）
                        display_counts: bool = False,
                        class_order: Optional[list] = None,
                        q_counts_vec: Optional[np.ndarray] = None,
                        c_counts_mat: Optional[np.ndarray] = None,
                        name2idx_for_counts: Optional[dict] = None,
                        # === 新增：overlay 相關 ===
                        overlay_nodes_2d: bool = False,
                        proj_intrin_query: str = "real",
                        proj_intrin_cand: str = "syn",
                        q_graph: any = None,
                        cand_name2graph: Optional[Dict[str, any]] = None,
                        overlay_fontscale: float = 0.0,
                        ref_item: Optional[dict] = None,
                        debug_q_overlay: bool = False,
                        out_root: Optional[Path] = None,
                        ):
    os.makedirs(os.path.dirname(out_png), exist_ok=True,)

    cell_h = 180
    # 顯示 counts 時預設更寬
    if cell_w_override is not None:
        cell_w = cell_w_override
    else:
        cell_w = 560 if display_counts else (int(cell_h * 2.0) if label_classes else int(cell_h * 4 / 3))

    title_h = 36

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font_s = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_m = ImageFont.truetype("DejaVuSansMono.ttf", 12)  # 等寬字顯示 vector
    except Exception:
        font = ImageFont.load_default()
        font_b = ImageFont.load_default()
        font_s = ImageFont.load_default()
        font_m = ImageFont.load_default()

    # ---- 動態估計底部高度（要能容納 2x2 layout 的四行/邊）----
    has_layout_text = (q_layout_vec is not None) and (c_layout_mat is not None)

    if display_counts:
        base_foot = 18 + 18 + 18 + 18  # rank/score + classes + Query[] + Candidate[]
    else:
        label_h = 16 if label_classes else 0
        strip_h = 12
        gap = 4
        n_groups = 2 if viz_node_mode == "both" else 1
        strips_total = n_groups * (strip_h * 2 + gap + 6)
        base_foot = 2 + 16 + (label_h + 4 if label_h else 0) + strips_total + 16

    if has_layout_text:
        # classes_line(16) + "Query patches:"(14) + 4行*14 + "Candidate patches:"(14) + 4行*14
        patch_rows = layout_rows * layout_cols  # 2x2 => 4
        layout_block = 16 + 14 + patch_rows * 14 + 14 + patch_rows * 14
        foot_h = max(base_foot, 18 + layout_block)
    else:
        foot_h = base_foot

    extra_iou_line = any(isinstance(it, dict) and ("pc_iou" in it) and (it["pc_iou"] is not None)
                     for it in top_ref_items)
    if extra_iou_line:
        foot_h += 18

    q_path = _resolve_img_path(query_item, query_root)
    q_im = _safe_open_image(q_path, target_h=cell_h)

    # Query overlay（先畫在 q_im 再貼到 canvas）
    if q_graph is not None:
        uvs_q = project_nodes_to_uvs(q_graph, intrinsics=proj_intrin_query,
                                     valid_classes=STRUCTURAL_CLASSES)
        
        # PIL 版 overlay（保持與現有 PIL 繪製管線一致）
        try:
            q_im = overlay_nodes_2d_on_tile_PIL(q_im, uvs_q, STRUCTURAL_CLASSES,
                                                img_w=IMG_W, img_h=IMG_H, font=font)
            if debug_q_overlay:
                dbg_dir = Path(out_root) / "__debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)

                # 1) 給 q_im 打上「顯眼水印＋邊框」，以便判斷 header 是否真的使用 override
                q_im_dbg = q_im.copy()
                draw = ImageDraw.Draw(q_im_dbg)
                # 邊框
                w, h = q_im_dbg.size
                draw.rectangle([(1,1), (w-2, h-2)], outline=(255, 0, 255), width=6)
                # 角落水印
                draw.text((12, 12), "Q_OVR", fill=(255, 0, 255))

                # 2) 存檔：header 前的 override 素材
                q_stem = Path(q_path).stem
                (dbg_dir / f"{q_stem}_q_im_override_before_header.png").unlink(missing_ok=True)
                q_im_dbg.save(dbg_dir / f"{q_stem}_q_im_override_before_header.png")

                # 3) 用 debug 版當 override（最直觀）
                q_im_for_header = q_im_dbg
            else:
                q_im_for_header = q_im

        except NameError:
            # 若使用者尚未引入 PIL 版 overlay，則靜默忽略 overlay
            pass

    # 1) 先組 header（不覆蓋圖片）
    ref_path = ref_item.get("path") if (ref_item is not None) else None
    if ref_path is not None:
        ref_path = _resolve_img_path(ref_path, cand_root)
    ref_name = ref_item.get("name", "ref") if (ref_item is not None) else "ref"
    ref_score = ref_item.get("score") if (ref_item is not None) else None
    ref_iou = ref_item.get("pc_iou") if (ref_item is not None) else None

    header_img, header_w, header_h = build_query_ref_header(
        q_img_path=str(q_path),
        q_name=Path(q_path).name,
        ref_img_path=ref_path,
        ref_name=ref_name,
        ref_score=ref_score,
        ref_iou=ref_iou,
        cell_h=cell_h,
        pad=pad,
        font=font,
        font_b=font_b,
        text_panel_min_w=220,  # 想要更寬可調 260/300
        q_im_override=q_im_for_header,
    )

    q_w = q_im.size[0]

    grid_w = cols * (cell_w + pad) + pad
    grid_h = rows * (cell_h + foot_h + pad) + pad
    W = max(grid_w, header_w)
    H = title_h + pad + header_h + pad + grid_h

    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    

    draw.text((pad, 8), "Top results (paged)", font=font_b, fill=(0, 0, 0))
    canvas.paste(header_img, (0, title_h + pad))

    start_y = title_h + pad + header_h + pad

    # 預先格式化類別標頭（每格都要印）
    classes_line = None
    if display_counts and class_order is not None:
        # 依需求含引號
        classes_line = "[ " + ", ".join([f"\"{c}\"" for c in class_order]) + " ]"

    for i in range(min(len(top_ref_items), cols * rows)):
        r = i // cols; c = i % cols
        x = pad + c * (cell_w + pad)
        y = start_y + r * (cell_h + foot_h + pad)

        item = top_ref_items[i]
        c_path = _resolve_img_path(item, cand_root)
        im = _safe_open_image(c_path, target_h=cell_h)

        # Candidate overlay（先畫在 im 再貼到 canvas）
        if overlay_nodes_2d and (cand_name2graph is not None):
            cn_for_graph = item["name"] if (isinstance(item, dict) and "name" in item) else Path(c_path).name
            g = _lookup_graph_by_name(cn_for_graph, cand_name2graph) if cand_name2graph else None
            if g is not None:
                uvs_c = project_nodes_to_uvs(g, intrinsics=proj_intrin_cand,
                                             valid_classes=STRUCTURAL_CLASSES)
                try:
                    im = overlay_nodes_2d_on_tile_PIL(im, uvs_c, STRUCTURAL_CLASSES,
                                                      img_w=IMG_W, img_h=IMG_H, font=font)
                except NameError:
                    pass

        im_x = x + (cell_w - im.size[0]) // 2
        canvas.paste(im, (im_x, y))

        rank = rank_offset + i + 1
        nm = Path(c_path).name
        nm_disp = (nm[:25] + "…") if len(nm) > 28 else nm

        # 取 candidate index（為了 counts）
        cand_name = item["name"] if isinstance(item, dict) and "name" in item else nm
        cidx = None
        if name2idx_for_counts is not None:
            cidx = name2idx_for_counts.get(cand_name, name2idx_for_counts.get(Path(cand_name).name))

        base_y = y + cell_h + 2
        # 預設 rank/score 一行（score 可留空）
        score_val = item.get("score") if isinstance(item, dict) else None
        score_txt = f"{float(score_val):.4f}" if isinstance(score_val, (int,float,np.floating)) else "-"
        draw.text((x, base_y), f"#{rank}  score={score_txt}", font=font, fill=(10,10,10))
        base_y += 18
        if isinstance(item, dict) and ("pc_iou" in item) and (item["pc_iou"] is not None):
            draw.text((x, base_y), f"IoU(pc) = {item['pc_iou']:.3f}", font=font, fill=(10,10,10))
            base_y += 18

         # === 印 2×2 layout counts（每個 patch 一行） ===
        if (q_layout_vec is not None) and (c_layout_mat is not None):
            # 先找 candidate 的 index：優先 cand_name2idx，否則回退用上面 counts 的 cidx
            cidx_layout = None
            if cand_name2idx is not None:
                cidx_layout = cand_name2idx.get(cand_name, cand_name2idx.get(Path(cand_name).name))
            if cidx_layout is None:
                cidx_layout = cidx  # ← 回退，用 counts 的映射

            if cidx_layout is not None and 0 <= int(cidx_layout) < c_layout_mat.shape[0]:
                classes_for_layout = (layout_classes or STRUCTURAL_CLASSES)
                # 類別順序一行
                classes_line_local = "[ " + ", ".join([f"\"{cname}\"" for cname in classes_for_layout]) + " ]"
                draw.text((x, base_y), classes_line_local, font=font_s, fill=(30, 30, 30))
                base_y += 16

                def _fmt_patches(vec_flat, rows_, cols_, C_):
                    v = np.asarray(vec_flat).reshape(-1)              # 攤平成 1-D
                    m = v.reshape(rows_ * cols_, C_).astype(int)      # (4, C)
                    lines_ = []
                    for pi in range(rows_ * cols_):
                        # 每個 patch 一行，避免太擠不再印 (r,c) 前綴
                        lines_.append("[" + ",".join(str(vv) for vv in m[pi].tolist()) + "]")
                    return lines_

                Cn = len(classes_for_layout)

                # Query 四行
                q_lines = _fmt_patches(q_layout_vec, layout_rows, layout_cols, Cn)
                draw.text((x, base_y), "Query patches:", font=font_s, fill=(20, 20, 20)); base_y += 14
                for ln in q_lines:
                    draw.text((x, base_y), ln, font=font_m, fill=(20, 20, 20)); base_y += 14

                # Candidate 四行
                c_vec = c_layout_mat[int(cidx_layout), :]
                c_lines = _fmt_patches(c_vec, layout_rows, layout_cols, Cn)
                draw.text((x, base_y), "Candidate patches:", font=font_s, fill=(20, 20, 20)); base_y += 14
                for ln in c_lines:
                    draw.text((x, base_y), ln, font=font_m, fill=(20, 20, 20)); base_y += 14


        if display_counts and class_order is not None and q_counts_vec is not None and c_counts_mat is not None and cidx is not None:
            # 類別標頭
            # draw.text((x, base_y), classes_line, font=font_s, fill=(30,30,30))
            base_y += 18

            # Query: [ ... ] （整數）
            q_counts = q_counts_vec.astype(int).tolist()
            q_line = "Query:     [" + ",".join(str(v) for v in q_counts) + "]"
            draw.text((x, base_y), q_line, font=font_m, fill=(20,20,20))
            base_y += 18

            # Candidate: [ ... ]
            c_counts = c_counts_mat[cidx, :].astype(int).tolist()
            c_line = "Candidate: [" + ",".join(str(v) for v in c_counts) + "]"
            draw.text((x, base_y), c_line, font=font_m, fill=(20,20,20))
            base_y += 18
        else:
            # 沒開 counts 顯示就走你原本的（條帶/類別）分支…（略）
            pass

        draw.text((x, base_y), nm_disp, font=font, fill=(60,60,60))

    canvas.save(out_png)

def visualize_grid_pages(query_item,
                         top_refs,
                         out_dir: Path,
                         base_name: str,
                         query_root: Optional[Path] = None,
                         cand_root: Optional[Path] = None,
                         page_size: int = None,
                         # 既有參數（略）...
                         cols: Optional[int] = None,
                         rows: Optional[int] = None,  
                         cell_w_override: Optional[int] = None,
                         # ★ 新增：counts 顯示所需
                         display_counts: bool = False,
                         class_order: Optional[list] = None,
                         q_counts_vec: Optional[np.ndarray] = None,
                         c_counts_mat: Optional[np.ndarray] = None,
                         name2idx_for_counts: Optional[dict] = None,
                         q_layout_vec: Optional[np.ndarray] = None,      # 該 query 的 layout 向量 (4*9,)
                        c_layout_mat: Optional[np.ndarray] = None,      # 全部 candidates 的 layout 矩陣 (Nc, 4*9)
                        layout_rows: int = LAYOUT_ROWS,                 # 固定 2
                        layout_cols: int = LAYOUT_COLS,                 # 固定 2
                        layout_classes: Optional[List[str]] = None,      # 類別順序；預設用 LAYOUT_CLASSES
                        overlay_nodes_2d: bool = False,
                         proj_intrin_query: str = "real",
                         proj_intrin_cand: str = "syn",
                         q_graph: any = None,
                         cand_name2graph: Optional[Dict[str, any]] = None,
                         overlay_fontscale: float = 0.0,
                         ref_item=None,
                         debug_q_overlay: bool = False,
                         out_root: Optional[Path] = None,
                         ):
    os.makedirs(out_dir, exist_ok=True)
    # 預設：開 counts → 用 5 欄；否則 10 欄
    _cols = cols if cols is not None else (5 if display_counts else 10)
    _rows = 1
    if page_size is None:
        page_size = _cols * _rows

    n = len(top_refs)
    if n == 0: return
    pages = (n + page_size - 1) // page_size
    for p in range(pages):
        start = p * page_size
        end   = min((p + 1) * page_size, n)
        page_refs = top_refs[start:end]
        out_png = out_dir / f"{base_name}_{start+1}-{end}.png"
        visualize_grid_page(
            query_item=query_item,
            top_ref_items=page_refs,
            out_png=str(out_png),
            rank_offset=start,
            query_root=query_root,
            cand_root=cand_root,
            cols=_cols, rows=_rows, pad=8,
            # …原本透傳的參數（略）…
            # counts 相關
            display_counts=display_counts,
            class_order=class_order,
            q_counts_vec=q_counts_vec,
            c_counts_mat=c_counts_mat,
            name2idx_for_counts=name2idx_for_counts,
            # 開 counts 預設加寬（若呼叫有給 cell_w_override 則以覆寫為準）
            cell_w_override=(cell_w_override if cell_w_override is not None
                             else (560 if display_counts else None)),
            q_layout_vec=q_layout_vec,          # 這張 query 的 layout (4*9,)
            c_layout_mat=c_layout_mat,          # 所有 candidates 的 layout (Nc, 4*9)
            layout_rows=layout_rows,            # =2
            layout_cols=layout_cols,            # =2
            layout_classes=(layout_classes or STRUCTURAL_CLASSES),
            overlay_nodes_2d=overlay_nodes_2d,
            proj_intrin_query=proj_intrin_query,
            proj_intrin_cand=proj_intrin_cand,
            q_graph=q_graph,
            cand_name2graph=cand_name2graph,
            overlay_fontscale=overlay_fontscale,
            ref_item=ref_item
        )

# ---------------- Rankers ----------------
def rank_node_only(q_desc, cand_descs, k, use_mask, node_len):
    q = q_desc[:node_len]
    C = cand_descs[:, :node_len]
    sims, idx = cosine_topk_choose(q, C, k, use_mask=use_mask)
    return sims[0], idx[0]

def rank_node_only_from_mats(q_node_vec, c_node_mat, k, use_mask):
    sims, idx = cosine_topk_choose(q_node_vec, c_node_mat, k, use_mask=use_mask)
    return sims[0], idx[0]

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

def rank_node_attr_chi2(q_attr_vec, c_attr_mat, k, tau=25.0):
    """
    用 Chi-square 距離對 node-attribute 向量排名。
    - 距離越小越像；若 tau>0 則回傳相似度 score=exp(-d/tau)，否則 score=-distance
    """
    dists = chi2_distance_all(q_attr_vec, c_attr_mat)  # 小→像
    Nc = dists.shape[0]
    k_eff = min(k, Nc)
    part = np.argpartition(dists, k_eff - 1)[:k_eff]
    idx_sorted = part[np.argsort(dists[part])]
    if tau and tau > 0:
        scores = np.exp(-dists[idx_sorted] / float(tau))
    else:
        scores = -dists[idx_sorted]
    return scores.astype(np.float32), idx_sorted.astype(int)

def rank_edge_only(q_desc, cand_descs, k, use_mask, node_len):
    q = q_desc[node_len:]
    C = cand_descs[:, node_len:]
    sims, idx = cosine_topk_choose(q, C, k, use_mask=use_mask)
    return sims[0], idx[0]

def rank_coarse(q_desc, cand_descs, k, use_mask, node_len, N_sim=1.0, E_sim=1.0):
    s_n, i_n = rank_node_only(q_desc, cand_descs, k=len(cand_descs), use_mask=use_mask, node_len=node_len)
    s_e, i_e = rank_edge_only(q_desc, cand_descs, k=len(cand_descs), use_mask=use_mask, node_len=node_len)
    # 對齊同一個候選集合（直接用全庫對應 index）
    # s_n/e 現在是針對全庫排列好的分數，但我們只需要融合後的整體排序
    # 直接建立分數陣列（長度 Nc），未出現在各自 top 時會是預設值（極小）
    Nc = cand_descs.shape[0]
    arr_n = np.full(Nc, -1.0, dtype=np.float32)
    arr_e = np.full(Nc, -1.0, dtype=np.float32)
    arr_n[i_n] = s_n
    arr_e[i_e] = s_e
    s_comb = (N_sim * arr_n + E_sim * arr_e) / max(1e-8, (N_sim + E_sim))
    # 取前 k
    k_eff = min(k, Nc)
    part = np.argpartition(-s_comb, k_eff - 1)[:k_eff]
    order = np.argsort(-s_comb[part])
    idx = part[order]
    return s_comb[idx], idx

def rank_coarse_with_node_scores(q_node_vec, c_node_mat, q_desc, c_descs, k,
                                 node_metric: str, use_mask: bool, tau: float,
                                 N_sim=1.0, E_sim=1.0):
    # node: 取得全庫分數（已是「高→好」）
    arr_n = node_scores_all(q_node_vec, c_node_mat, node_metric, use_mask, tau)
    # edge: 仍用原本 full-descriptor 的 edge 子向量
    C = len(STRUCTURAL_CLASSES)
    node_len_full = C * (1 + STAT_PER_CLASS)
    q_edge = q_desc[node_len_full:]
    C_edge = c_descs[:, node_len_full:]
    q = q_edge; Cmat = C_edge
    if use_mask:
        m = q != 0
        if np.any(m):
            q = q[m]; Cmat = Cmat[:, m]
    qn = np.linalg.norm(q) + 1e-8
    Cn = np.linalg.norm(Cmat, axis=1) + 1e-8
    arr_e = ((Cmat @ q) / (Cn * qn)).astype(np.float32)

    s_comb = (N_sim * arr_n + E_sim * arr_e) / max(1e-8, (N_sim + E_sim))
    Nc = s_comb.shape[0]; k_eff = min(k, Nc)
    part = np.argpartition(-s_comb, k_eff - 1)[:k_eff]
    idx = part[np.argsort(-s_comb[part])]
    return s_comb[idx], idx

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

def rank_ged_only(q_idx, q_graphs, cand_graphs, pool_idx, k):
    """
    只用 GED 成本打分（越小越好）。回傳 score=-ged_cost。
    pool_idx: 先前粗排結果的候選池索引
    """
    scores = []
    idxs = []
    for i in pool_idx:
        try:
            cost = ged_cost(q_graphs[q_idx], cand_graphs[i])
        except Exception:
            cost = float('inf')
        sc = -float(cost)  # 依你的設定，score = -ged_cost
        scores.append(sc)
        idxs.append(i)
    scores = np.asarray(scores, dtype=np.float32)
    idxs = np.asarray(idxs, dtype=int)
    if scores.size == 0:
        return scores, idxs
    k_eff = min(k, scores.size)
    part = np.argpartition(-scores, k_eff - 1)[:k_eff]
    order = np.argsort(-scores[part])
    idx_sel = part[order]
    return scores[idx_sel], idxs[idx_sel]

def rank_by_ged_all_candidates(
    qg: nx.Graph,
    cand_graphs: List[nx.Graph],
    top_k: int,
    timeout: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    對「所有」candidate graphs 計算 GED 並排名（不做任何 pool 篩選）。
    依照 graph_bear_GED_0918 的成本設定：
      - node_subst_cost = node_cost（需預先定義）
      - edge_subst_cost = edge_cost（需預先定義）
      - node_del_cost   = LAMBDA_NODE
      - edge_del_cost   = LAMBDA_EDGE
      - node_ins_cost   = 0.0
      - edge_ins_cost   = 0.0
      - edge_match: 僅允許 relation 相同的邊做替換
    分數定義：score = -GED_cost（成本越小分數越高）

    參數
    ----
    qg : nx.Graph
        Query graph
    cand_graphs : List[nx.Graph]
        候選圖清單（全部會被計算）
    top_k : int
        取前 K 名
    timeout : float
        傳遞給 networkx.optimize_graph_edit_distance 的逾時秒數

    回傳
    ----
    scores : np.ndarray, shape = (Nc,)
        對所有 Nc 個 candidates 的分數（= -GED_cost）
    idxs : np.ndarray, shape = (min(top_k, Nc),)
        Top-K 之 candidates 的索引（依分數由高到低）
    """
    Nc = len(cand_graphs)
    if Nc == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    scores = np.full((Nc,), -np.inf, dtype=np.float32)

    for i, cg in enumerate(cand_graphs):
        try:
            e_iter = nx.optimize_graph_edit_distance(
                qg, cg,
                node_subst_cost=node_cost,
                node_del_cost=lambda n: LAMBDA_NODE,
                node_ins_cost=lambda n: 0.0,
                edge_subst_cost=edge_cost,
                edge_del_cost=lambda e: LAMBDA_EDGE,
                edge_ins_cost=lambda e: 0.0,
                edge_match=lambda e1, e2: e1.get('relation') == e2.get('relation'),
                timeout=timeout,
            )
            # 速度導向：取第一個可行上界（與原檔一致）
            cost = next(e_iter)
            scores[i] = -float(cost)  # 分數越大越好
        except StopIteration:
            # 無可行解（或逾時前沒產生解），視為極差
            scores[i] = -float('inf')
        except Exception:
            # 任何計算異常直接視為極差
            scores[i] = -float('inf')

    k = min(top_k, Nc)
    if k == Nc:
        idxs_sorted = np.argsort(scores)[::-1]
    else:
        # 先用 argpartition 取前 k 名，再在子集合中做遞減排序
        topk_unsorted = np.argpartition(scores, -k)[-k:]
        idxs_sorted = topk_unsorted[np.argsort(scores[topk_unsorted])[::-1]]

    return scores, idxs_sorted







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

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    an = float(np.linalg.norm(a)); bn = float(np.linalg.norm(b))
    if an < eps or bn < eps: return float("nan")
    return float(np.dot(a, b) / ((an + eps) * (bn + eps)))

def _tokens(arr: np.ndarray, decimals: int):
    return [f"{v:.{decimals}f}" for v in arr.tolist()]

def _draw_wrapped_tokens(draw, x, y, width, tokens, font, line_gap=2, prefix=None):
    """
    將 tokens 以逗號分隔、在指定寬度內自動換行，回傳使用掉的高度（像素）。
    若給 prefix（如 'Q:'/'C:'），會在行首加上。
    """
    h = 0
    line = (prefix + " ") if prefix else ""
    sep = ", "
    for t in tokens:
        cand = (line + (sep if line.strip() else "") + t)
        try:
            wpx = draw.textlength(cand, font=font)
        except Exception:
            wpx = font.getsize(cand)[0]
        if wpx <= width:
            line = cand
        else:
            draw.text((x, y + h), line, font=font, fill=(30,30,30))
            h += font.size + line_gap
            line = (prefix + " " + t) if prefix else t
    if line:
        draw.text((x, y + h), line, font=font, fill=(30,30,30))
        h += font.size + line_gap
    return h

def node_scores_all(q_node_vec, c_node_mat, metric: str, use_mask: bool, tau: float = 0.0):
    if metric == "chi2":
        q = q_node_vec
        C = c_node_mat
        if use_mask:
            m = q > 0
            if np.any(m):
                q = q[m]; C = C[:, m]
        d = chi2_distance_all(q, C)      # 小→像
        return (np.exp(-d / tau) if (tau and tau > 0) else -d).astype(np.float32)
    else:  # cosine
        q = np.asarray(q_node_vec, dtype=np.float32)
        C = np.asarray(c_node_mat, dtype=np.float32)
        if use_mask:
            m = q != 0
            if np.any(m):
                q = q[m]; C = C[:, m]
        qn = np.linalg.norm(q) + 1e-8
        Cn = np.linalg.norm(C, axis=1) + 1e-8
        sims = (C @ q) / (Cn * qn)
        return sims.astype(np.float32)

def rank_node_counts_chi2(q_counts_vec, c_counts_mat, k, mask_query_nonzero=False, tau=LAYOUT_TAU):
    q = np.asarray(q_counts_vec, dtype=np.float32).ravel()
    C = np.asarray(c_counts_mat, dtype=np.float32)
    # 可選遮罩：僅用 query>0 的類別（如果全部為 0 就用全部）
    if mask_query_nonzero:
        m = q > 0
        if np.any(m):
            q = q[m]; C = C[:, m]
    dists = chi2_distance_all(q, C)  # 小→像
    # 取最小的 k
    Nc = dists.shape[0]
    k_eff = min(k, Nc)
    part = np.argpartition(dists, k_eff - 1)[:k_eff]
    idx_sorted = part[np.argsort(dists[part])]
    # 轉 score（高→好）
    if tau and tau > 0:
        scores = np.exp(-dists[idx_sorted] / float(tau))
    else:
        scores = -dists[idx_sorted]  # 與 GED 一致：score = -distance
    return scores.astype(np.float32), idx_sorted.astype(int)

def get_node_mats_by_mode(q_descs, c_descs, q_graphs, c_graphs, mode: str, class_order):
    if mode == "counts":
        q_node_mat = build_counts_mat(q_graphs, class_order)   # (Nq,C)
        c_node_mat = build_counts_mat(c_graphs, class_order)   # (Nc,C)
        node_len = len(class_order)
    elif mode == "attr":
        # 新增：node-attribute（每類5維 => 總長 = C*5）
        q_node_mat = build_node_attr_mat(q_graphs, class_order)    # (Nq, C*5)
        c_node_mat = build_node_attr_mat(c_graphs, class_order)    # (Nc, C*5)
        node_len = len(class_order) * 5
    elif mode == "seq":
        return None, None, 0
    else:  # full/descriptor
        q_node_mat, _, node_len = split_node_edge_desc(q_descs)
        c_node_mat, _, _        = split_node_edge_desc(c_descs)
    return q_node_mat, c_node_mat, node_len

def rank_node_layout_chi2(q_layout_vec: np.ndarray,
                          c_layout_mat: np.ndarray,
                          k: int,
                          tau: float = 250,
                          eps: float = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    用 2x2xC 的 layout 向量做 Chi-square 距離，分數=exp(-chi2/tau)，回傳 (scores_topk, idxs_topk)。
    """
    q = np.asarray(q_layout_vec, dtype=np.float32).reshape(1, -1)   # (1, 4*C)
    C = np.asarray(c_layout_mat, dtype=np.float32)                  # (Nc, 4*C)

    # --------Chi-square------------
    # diff  = C - q
    # denom = C + q + eps
    # chi2  = 0.5 * np.sum((diff * diff) / denom, axis=1, dtype=np.float64).astype(np.float32)

    # sim   = np.exp(-chi2 / float(tau)).astype(np.float32)           # 越大越像

    # ----------chi² augmented----------
    q  = q / max(q.sum(), 1e-6)
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-6)

    diff  = Cn - q
    denom = Cn + q + 1e-8
    chi2  = 0.5 * np.sum((diff * diff) / denom, axis=1).astype(np.float32)

    tau = max(np.median(chi2), 1e-6)         # 自動定標
    sim = np.exp(-chi2 / tau).astype(np.float32)

    # Top-k（降序）
    k_eff = max(1, min(int(k), sim.shape[0]))
    part = np.argpartition(-sim, k_eff - 1)[:k_eff]
    idx_sorted = part[np.argsort(-sim[part])]
    return sim[idx_sorted], idx_sorted

def rerank_topk_by_layout(q_layout_vec: np.ndarray,
                          c_layout_mat: np.ndarray,
                          idxs_topk: np.ndarray,
                          tau: float = 1.0,
                          eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    僅在 idxs_topk 指定的子集中，用 layout(2x2xC) 做 χ² → exp(-d/τ) 重排。
    回傳：(scores_ranked, idxs_ranked, chi2_ranked)
    """
    q = np.asarray(q_layout_vec, dtype=np.float32).reshape(1, -1)
    subset = c_layout_mat[idxs_topk, :].astype(np.float32)   # (K, 4*C)

    diff  = subset - q
    denom = subset + q + eps
    chi2  = 0.5 * np.sum((diff * diff) / denom, axis=1, dtype=np.float64).astype(np.float32)

    tau_safe = float(max(tau, 1e-6))
    sim  = np.exp(-chi2 / tau_safe).astype(np.float32)       # 越大越像

    order = np.argsort(-sim)                                  # 在 K 個裡排序
    return sim[order], idxs_topk[order], chi2[order]

def build_counts_mat(graphs, class_order):
    """
    依 class_order 回傳 (N, C) 計數矩陣（float32，但都是整數值）。
    只統計 class_order 中的類別；不在清單的類別一律忽略。
    """
    C = len(class_order)
    idx = {cls: i for i, cls in enumerate(class_order)}
    M = np.zeros((len(graphs), C), dtype=np.float32)
    for gi, g in enumerate(graphs):
        for _, data in g.nodes(data=True):
            cat = data.get("category")
            if cat in idx:
                M[gi, idx[cat]] += 1.0
    return M

def _build_node_attr_one(g, class_order):
    """
    回傳單一 graph 的 node-attribute 向量：長度 = len(class_order)*5，
    依序為每類別的 [w_mean, h_mean, area_mean(log1p 已處理), depth_mean, depth_std]。
    規則：
      - 缺任一欄位(w/h/area/depth)的 node 直接忽略
      - depth_std 用母體標準差(ddof=0)；只有1筆資料 -> 0
      - area_mean 先做 log1p 後再併入
      - 向量最後做 L1 normalize（若總和=0則不處理）
    """
    import numpy as np

    feats_per_class = 5
    vec = np.zeros((len(class_order) * feats_per_class,), dtype=np.float32)
    cls2idx = {cls: i for i, cls in enumerate(class_order)}

    # 先將各類別的值累積起來
    buckets = {cls: {"w": [], "h": [], "area": [], "depth": []} for cls in class_order}
    for _, data in g.nodes(data=True):
        cat = data.get("category")
        if cat not in buckets:
            continue
        # 缺的忽略：只要四者有一個缺，就跳過
        if not all(k in data for k in ("w", "h", "area", "depth")):
            continue
        try:
            w = float(data["w"]); h = float(data["h"])
            area = float(data["area"]); depth = float(data["depth"])
        except Exception:
            # 轉型失敗也視為缺資料 => 忽略
            continue
        buckets[cat]["w"].append(w)
        buckets[cat]["h"].append(h)
        buckets[cat]["area"].append(area)
        buckets[cat]["depth"].append(depth)

    # 逐類別做平均/標準差並寫回向量
    for cls, i_cls in cls2idx.items():
        w_list = buckets[cls]["w"]
        h_list = buckets[cls]["h"]
        a_list = buckets[cls]["area"]
        d_list = buckets[cls]["depth"]

        base = i_cls * feats_per_class
        if len(w_list) == 0 or len(h_list) == 0 or len(a_list) == 0 or len(d_list) == 0:
            # 該類別無有效 node => 該5維維持 0
            continue

        w_mean = float(np.mean(w_list))
        h_mean = float(np.mean(h_list))
        area_mean = float(np.mean(a_list))
        depth_mean = float(np.mean(d_list))
        depth_std = float(np.std(d_list, ddof=0))  # 單筆 => 0

        # 對 area 做 log1p 穩定尺度
        area_mean = float(np.log1p(max(area_mean, 0.0)))

        vec[base + 0] = w_mean
        vec[base + 1] = h_mean
        vec[base + 2] = area_mean
        vec[base + 3] = depth_mean
        vec[base + 4] = depth_std

    # L1 normalize（避免被單一量綁架；和=0 時不動）
    s = float(np.sum(np.abs(vec)))
    if s > 0:
        vec /= s
    return vec.astype(np.float32)

def build_node_attr_mat(graphs, class_order):
    """
    批次將 graph 轉為 (N, len(class_order)*5) 的 node-attribute 矩陣。
    """
    mats = [_build_node_attr_one(g, class_order) for g in graphs]
    return np.stack(mats, axis=0).astype(np.float32)




def chi2_distance_all(q_vec: np.ndarray, C_mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    q_vec: (D,), C_mat: (Nc,D)
    回傳每個 candidate 與 q_vec 的 χ² 距離（越小越像）。
    定義: 0.5 * Σ ( (c - q)^2 / (c + q + eps) )
    """
    q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
    C = np.asarray(C_mat, dtype=np.float32)
    diff = C - q          # (Nc,D)
    denom = C + q + eps   # (Nc,D)
    d = 0.5 * np.sum((diff * diff) / denom, axis=1, dtype=np.float64)
    return d.astype(np.float32)

def chi2_distance_pair(q_vec: np.ndarray, c_vec: np.ndarray, eps: float = 1e-8) -> float:
    q = np.asarray(q_vec, dtype=np.float32)
    c = np.asarray(c_vec, dtype=np.float32)
    diff = q - c
    denom = q + c + eps
    return float(0.5 * np.sum((diff * diff) / denom, dtype=np.float64))

def _pick_intrinsics(which: str) -> Tuple[float,float,float,float]:
    # which in {"real","syn","auto"}；你要強制 real/syn，所以這裡只處理 real/syn
    if which == "syn":
        return FX_SYN, FY_SYN, CX_SYN, CY_SYN
    return FX_REAL, FY_REAL, CX_REAL, CY_REAL

def project_nodes_to_uvs(graph,
                         intrinsics: str = "real",
                         valid_classes: List[str] = None) -> List[Tuple[float,float,str]]:
    """讀 graph.nodes 的 (cx,cy,cz,category) → 投影 (u,v,cat)。Z 無效直接跳過。"""
    if valid_classes is None:
        valid_classes = STRUCTURAL_CLASSES
    fx, fy, cx, cy = _pick_intrinsics(intrinsics)
    out: List[Tuple[float,float,str]] = []
    for _, data in graph.nodes(data=True):
        cat = data.get("category")
        if cat not in valid_classes: 
            continue
        if not all(k in data for k in ("cx","cy","cz")):
            continue
        uv = _project_uv((data["cx"], data["cy"], data["cz"]), fx, fy, cx, cy)
        if uv is None:
            continue   # Z 無效跳過
        out.append((uv[0], uv[1], cat))
    return out

def _overlay_text_with_bg_PIL(img_rgba: Image.Image, text: str, org: Tuple[int,int],
                              color: Tuple[int,int,int], font) -> None:
    """在 RGBA 圖上畫半透明底再寫字（強韌版：夾邊＋保證 x0<=x1,y0<=y1）。"""
    draw = ImageDraw.Draw(img_rgba)
    x = int(org[0]); y = int(org[1])

    # 先把文字起點夾進圖內，避免完全跑出界
    x = max(0, min(img_rgba.width  - 1, x))
    y = max(0, min(img_rgba.height - 1, y))

    # 估 bbox
    try:
        bbox = draw.textbbox((x, y), text, font=font)  # (left, top, right, bottom)
    except Exception:
        w, h = draw.textsize(text, font=font)
        bbox = (x, y - h, x + w, y)

    # padding + 夾邊
    pad = 3
    x0 = bbox[0] - pad; y0 = bbox[1] - pad
    x1 = bbox[2] + pad; y1 = bbox[3] + pad
    x0 = max(0, min(img_rgba.width  - 1, x0))
    y0 = max(0, min(img_rgba.height - 1, y0))
    x1 = max(0, min(img_rgba.width  - 1, x1))
    y1 = max(0, min(img_rgba.height - 1, y1))

    # ★ 關鍵：保證順序正確＋至少 1px 厚度
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    if x1 == x0: x1 = min(img_rgba.width  - 1, x0 + 1)
    if y1 == y0: y1 = min(img_rgba.height - 1, y0 + 1)

    # 畫半透明底
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 140))
    img_rgba.alpha_composite(overlay)

    # 寫字
    draw.text((x, y), text, fill=(color[0], color[1], color[2], 255), font=font)

def overlay_nodes_2d_on_tile_PIL(tile_rgb: Image.Image,
                                 uvs: List[Tuple[float,float,str]],
                                 classes: List[str],
                                 img_w: int = IMG_W,
                                 img_h: int = IMG_H,
                                 font=None,
                                 dot_radius: int = 4) -> Image.Image:
    """
    將 (u,v,category) 依 tile 尺寸縮放後畫在 tile_rgb 上（RGB→RGBA→RGB）。
    """
    # 轉 RGBA 避免覆蓋
    tile = tile_rgb.convert("RGBA")
    w, h = tile.size
    sx = w / float(img_w)
    sy = h / float(img_h)
    # 疊字位移
    offset_map: Dict[Tuple[int,int], int] = {}

    for (u, v, cat) in uvs:
        tx = int(round(u * sx)); ty = int(round(v * sy))
        tx = max(0, min(w-1, tx)); ty = max(0, min(h-1, ty))
        # 微偏移避免重疊
        key = (tx, ty)
        k = offset_map.get(key, 0); offset_map[key] = k + 1
        ox = tx + (k % 3) * 8
        oy = ty - (k // 3) * 10

        color = CLASS_COLORS.get(cat, (255,255,255))
        # 畫小圓點
        draw = ImageDraw.Draw(tile)
        draw.ellipse([tx - dot_radius, ty - dot_radius, tx + dot_radius, ty + dot_radius],
                     fill=color, outline=None)
        # 文字（半透明底）
        tx_text = max(0, min(w - 1, ox + 6))
        ty_text = max(0, min(h - 1, oy - 6))
        _overlay_text_with_bg_PIL(tile, cat, (tx_text, ty_text), color, font or ImageFont.load_default())



    return tile.convert("RGB")

def _lookup_graph_by_name(name: str, name2graph: Dict[str, any]) -> Optional[any]:
    if name in name2graph:
        return name2graph[name]
    b = Path(name).name
    return name2graph.get(b, None)


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
    parser = argparse.ArgumentParser(description="Simplified seq+wire demo")
    parser.add_argument('--query_json', type=Path, required=True)
    parser.add_argument('--candidate_json', type=Path, required=True)
    parser.add_argument('--query_root', type=Path, required=True, help="absolute folder for query images")
    parser.add_argument('--candidate_root', type=Path, required=True, help="absolute folder for candidate images")
    parser.add_argument('--output_json', type=Path, default=Path('Retrieval_results.json'))
    parser.add_argument('--out_root', type=Path, default=Path('./viz'), help='root folder for visualization/results')
    parser.add_argument('--keywords', nargs='+', default=[])
    parser.add_argument('--wire_query_json', type=Path, default=None, help='wireframe JSON for queries')
    parser.add_argument('--wire_candidate_json', type=Path, default=None, help='wireframe JSON for candidates')
    parser.add_argument('--wire_target_hw', nargs=2, type=int, metavar=('H','W'), default=None)
    parser.add_argument('--wire_grid', nargs=2, type=int, metavar=('rows','cols'), default=[4, 4])
    parser.add_argument('--wire_score_th', type=float, default=0.75)
    parser.add_argument('--wire_bins', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--fusion', choices=['score', 'rrf'], default='rrf')
    parser.add_argument('--w_seq', type=float, default=0.5)
    parser.add_argument('--w_wire', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--q_low', type=float, default=0.05)
    parser.add_argument('--q_high', type=float, default=0.95)
    parser.add_argument('--rrf_c', type=int, default=60)
    parser.add_argument('--coarse_output', type=Path, default=None,
                        help="MI-style JSON output: {'results': {query_img: [{'ref_image': cand_img}, ...]}}")
    args = parser.parse_args()

    q_names, q_graphs, q_descs = load_graphs(args.query_json)
    c_names, c_graphs, c_descs = load_graphs(args.candidate_json)

    if args.keywords:
        keep = []
        for name, g, d in zip(c_names, c_graphs, c_descs):
            if any(kw in name for kw in args.keywords):
                keep.append((name, g, d))
        if not keep:
            print(f"[!] No candidates match keywords {args.keywords}, exiting.")
            return
        c_names, c_graphs, c_descs = zip(*keep)
        c_descs = np.stack(c_descs, axis=0).astype(np.float32)
        print(f"[i] Filtered candidates down to {len(c_names)} using keywords {args.keywords}")

    class_order = list(STRUCTURAL_CLASSES)
    q_seq_cache = build_seq_xy_batch(q_graphs, class_order, intrinsics="real")
    c_seq_cache = build_seq_xy_batch(c_graphs, class_order, intrinsics="syn")

    q_name2img = name2imgpath_from_graphs(q_names, q_graphs)
    c_name2img = name2imgpath_from_graphs(c_names, c_graphs)

    wire_db_mat = None
    wire_db_name2idx = None
    wire_q_dict = None
    if args.wire_candidate_json and args.wire_query_json:
        db_entries = load_wire_json(args.wire_candidate_json)
        q_entries = load_wire_json(args.wire_query_json)
        if args.wire_target_hw is not None:
            H, W = map(int, args.wire_target_hw)
        elif q_entries:
            H = int(q_entries[0].get('height', 1080))
            W = int(q_entries[0].get('width', 1920))
        else:
            H, W = 1080, 1920
        rows, cols = map(int, args.wire_grid)
        score_th = float(args.wire_score_th)
        num_bins = int(args.wire_bins)
        q_basenames = {Path(p).name for p in q_name2img.values()}
        c_basenames = {Path(p).name for p in c_name2img.values()}
        wire_db_mat, wire_db_name2idx = build_wire_db_from_json(
            db_entries, H, W, rows, cols, score_th, num_bins, keep_basenames=c_basenames
        )
        wire_q_dict = build_wire_q_from_json(
            q_entries, H, W, rows, cols, score_th, num_bins, keep_basenames=q_basenames
        )

    out_dir = Path(args.out_root)
    out_collage = out_dir / "collages"
    out_json = out_dir / Path(args.output_json).name
    out_csv = out_dir / (Path(args.output_json).stem + ".csv")
    os.makedirs(out_collage, exist_ok=True)

    cand_name2graph = {name: g for name, g in zip(c_names, c_graphs)}
    for name, g in zip(c_names, c_graphs):
        cand_name2graph[Path(name).name] = g

    results = {}
    for qi, qn in enumerate(q_names):
        q_seq_xy = q_seq_cache[qi]
        Nc = len(c_seq_cache)
        seq_scores_full, idxs_full = rank_seq_lcs_xy(q_seq_xy, c_seq_cache, top_k=Nc, wx=0.5, wy=0.5)
        seq_scores = np.full(Nc, -np.inf, dtype=np.float32)
        seq_scores[idxs_full] = seq_scores_full

        wire_dists = np.full(Nc, np.inf, dtype=np.float32)
        if (wire_q_dict is not None) and (wire_db_mat is not None) and (wire_db_mat.size > 0):
            qb = Path(qn).name
            qv = wire_q_dict.get(qb, None)
            if qv is not None:
                wire_idx_of_c = np.array([wire_db_name2idx.get(Path(n).name, -1) for n in c_names], dtype=int)
                m = wire_idx_of_c >= 0
                if np.any(m):
                    d_all = chi2_distance_rowwise(qv, wire_db_mat[wire_idx_of_c[m], :])
                    wire_dists[m] = d_all.astype(np.float32)

        if args.fusion == 'rrf':
            fused = rrf_fuse_ranks(seq_scores, wire_dists, c=int(args.rrf_c))
        else:
            fused = fuse_scores(seq_scores, wire_dists, args)

        k_eff = min(int(args.top_k), Nc)
        part = np.argpartition(-fused, k_eff - 1)[:k_eff]
        order = np.argsort(-fused[part])
        idxs = part[order]
        scores = fused[idxs].astype(np.float32)

        rows = []
        for rank, ci in enumerate(idxs, 1):
            cn = c_names[ci]
            rows.append({
                "rank": rank,
                "name": cn,
                "score": float(scores[rank - 1]),
                "path": c_name2img.get(cn, cn),
                "seq_score": float(seq_scores[ci]) if np.isfinite(seq_scores[ci]) else None,
                "wire_chi2": float(wire_dists[ci]) if np.isfinite(wire_dists[ci]) else None,
            })
        results[qn] = rows

        visualize_grid_pages(
            query_item=qn,
            top_refs=rows,
            out_dir=out_collage,
            base_name=Path(qn).stem,
            query_root=args.query_root,
            cand_root=args.candidate_root,
            display_counts=False,
            class_order=class_order,
            q_counts_vec=None,
            c_counts_mat=None,
            name2idx_for_counts=None,
            q_layout_vec=None,
            c_layout_mat=None,
            overlay_nodes_2d=False,
            q_graph=q_graphs[qi],
            cand_name2graph=cand_name2graph,
        )

    if args.coarse_output:
        mi_results = {"results": {}}
        for qn in q_names:
            q_img_path = _resolve_img_path(qn, args.query_root)
            ref_list = []
            for row in results.get(qn, []):
                ref_path = _resolve_img_path(row["name"], args.candidate_root)
                ref_list.append({"ref_image": ref_path})
            mi_results["results"][str(q_img_path)] = ref_list
        out_coarse = str(args.coarse_output)
        os.makedirs(os.path.dirname(out_coarse), exist_ok=True)
        with open(out_coarse, "w", encoding="utf-8") as f:
            json.dump(mi_results, f, ensure_ascii=False, indent=2)
        print(f"[seq_wire] MI-style results saved to: {out_coarse}")

    write_results_json(str(out_json), results)
    write_results_csv(str(out_csv), results)
    print(f"[seq_wire] results saved to: {out_json} and {out_csv}")
    print(f"[seq_wire] collages saved to: {out_collage}")
if __name__ == '__main__':
    main()
