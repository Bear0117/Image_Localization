import json
import itertools
from pathlib import Path
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from matplotlib.patches import Rectangle
import random


def load_rgb(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB file not found: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_semantic(image_path: Path) -> np.ndarray:
    sem = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if sem is None:
        raise FileNotFoundError(f"Semantic file not found: {image_path}")
    # ensure RGBA
    if sem.ndim == 2:
        sem = cv2.cvtColor(sem, cv2.COLOR_GRAY2RGBA)
    if sem.shape[2] == 3:
        alpha = np.full(sem.shape[:2] + (1,), 255, dtype=np.uint8)
        sem = np.dstack((sem, alpha))
    return sem

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def polar_bins(vecs: np.ndarray, angle_bins: int, dist_bins: int):
    ang = (np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])) + 360.0) % 360.0
    dist = np.linalg.norm(vecs, axis=1)
    
    a_bin = (ang // (360.0 / max(1, angle_bins))).astype(int)
    a_bin = np.clip(a_bin, 0, angle_bins - 1)
    
    max_d = float(dist.max()) if dist.size > 0 else 0.0
    if max_d <= 1e-9:
        d_bin = np.zeros_like(a_bin, dtype=int)
    else:
        d_bin = (dist / max_d * dist_bins).astype(int)
        d_bin = np.clip(d_bin, 0, dist_bins - 1)

    return a_bin, d_bin



def node_bbox(g: nx.Graph, n: int, W: int, H: int, pad: float) -> np.ndarray:
    d = g.nodes[n]
    cx, cy = d['cx'] * W, d['cy'] * H
    bw, bh = d['w2d'] * W, d['h2d'] * H
    return np.array([cx - bw/2 - pad, cy - bh/2 - pad, cx + bw/2 + pad, cy + bh/2 + pad], np.float32)

def merge_by_iou(comps, category, W, H):
    group_list = [c for c in comps if c['category'] == category]
    others     = [c for c in comps if c['category'] != category]
    merged, used = [], set()
    for i, ci in enumerate(group_list):
        if i in used: continue
        bbox_i = ci['bbox2d']
        x0_i,y0_i,w_i,h_i = bbox_i
        x1_i, y1_i = x0_i+w_i, y0_i+h_i
        union_group = [ci]; used.add(i)
        mask_union  = ci['mask'].copy()
        for j, cj in enumerate(group_list[i+1:], start=i+1):
            if j in used: continue
            bbox_j = cj['bbox2d']
            x0_j,y0_j,w_j,h_j = bbox_j
            x1_j, y1_j = x0_j+w_j, y0_j+h_j
            # compute IoU
            inter_w = max(0, min(x1_i,x1_j) - max(x0_i,x0_j))
            inter_h = max(0, min(y1_i,y1_j) - max(y0_i,y0_j))
            if inter_w*inter_h == 0: continue
            area_i = w_i * h_i
            area_j = w_j * h_j
            iou = inter_w*inter_h / (area_i+area_j - inter_w*inter_h)
            if iou > 0:
                used.add(j)
                union_group.append(cj)
                mask_union |= cj['mask']
        # merge union_group
        xs = np.concatenate([[g['bbox2d'][0], g['bbox2d'][0]+g['bbox2d'][2]] for g in union_group])
        ys = np.concatenate([[g['bbox2d'][1], g['bbox2d'][1]+g['bbox2d'][3]] for g in union_group])
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w_, h_ = x1-x0, y1-y0
        merged.append({
            'category': category,
            'bbox': (x0, y0, w_, h_),
            'center': ((x0+w_/2)/W, (y0+h_/2)/H),
            'mask': mask_union
        })
    return merged, others



def load_synthetic_components(fn: str, RGB_FOLDER: Path, SEMANTIC_FOLDER: Path, ANNOTATION_FOLDER: Path, DEPTH_FOLDER: Path, CATEGORY_MAPPING: Dict[str, str], CAMERA_INTRINSICS: np.ndarray, STRUCTURAL_CLASSES: List[str]) -> tuple[np.ndarray, list[dict]]:
    rgb_path = RGB_FOLDER / fn
    sem_path = SEMANTIC_FOLDER / fn.replace('_rgb_image.png', '_ins_seg.png')
    ann_path = ANNOTATION_FOLDER / fn.replace('_rgb_image.png', '_ins_seg.json')
    depth_path = DEPTH_FOLDER / fn.replace("_rgb_image.png", "_distance_to_image_plane.npy")

    rgb = load_rgb(rgb_path)
    sem = load_semantic(sem_path)
    with open(ann_path, 'r') as f:
        ann_dict = json.load(f)
    depth_map = np.load(str(depth_path))
    H, W = rgb.shape[:2]


    components = []
    raw_comps = []
    colors = np.unique(sem.reshape(-1, 4), axis=0)

    fx, fy, Cx, Cy = CAMERA_INTRINSICS

    for color in colors:
        mask = np.all(sem == color, axis=-1)
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        y0, x0 = ys.min(), xs.min()
        y1, x1 = ys.max() + 1, xs.max() + 1
        bbox = (x0, y0, x1 - x0, y1 - y0)
        cx, cy = x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2

        # JSON key uses (R,G,B,A)
        key = f"({color[2]}, {color[1]}, {color[0]}, {color[3]})"
        full = ann_dict.get(key, '')
        parts = full.strip('/').split('/') if full else []
        label = parts[parts.index('Geometry')+1] if 'Geometry' in parts else (parts[3] if len(parts) >= 4 else 'Unknown')
        struct = CATEGORY_MAPPING.get(label)

        # ----------------------------
        # 可改為 Bbox Overlap
        # ----------------------------
        # Door 篩掉 Paint_DefaultNullMaterial
        # if struct == "Doors" and "Paint_DefaultNullMaterial" in full:
        #     continue
        # # Window 篩掉包含 896773
        # if struct == "Windows" and "896773" in full:
        #     continue


        if struct is None or struct not in STRUCTURAL_CLASSES:
            continue


        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        pts3d = []

        for u, v in corners:
            # clamp 到合法索引
            u_i = min(max(int(round(u)), 0), W - 1)
            v_i = min(max(int(round(v)), 0), H - 1)
            z = depth_map[v_i, u_i]
            X = (u - Cx) * z / fx
            Y = (v - Cy) * z / fy
            pts3d.append([X, Y, z])
        pts3d = np.array(pts3d)
        min3d, max3d = pts3d.min(0), pts3d.max(0)

        raw_comps.append({
            "category": struct,
            "bbox2d": bbox,
            "center2d": (cx/W, cy/H),
            "mask": mask,
            "min3d": min3d,
            "max3d": max3d
        })

        walls  = [c for c in raw_comps if c["category"]=="Walls"]
        others = [c for c in raw_comps if c["category"]!="Walls"]
        merged = []
        used = set()
        for i, wi in enumerate(walls):
            if i in used: continue
            group = [wi]; used.add(i)
            for j in range(i+1, len(walls)):
                if j in used: continue
                wj = walls[j]
                # 3D bbox 重疊檢查
                if np.all(wi["min3d"] <= wj["max3d"]) and np.all(wj["min3d"] <= wi["max3d"]):
                    group.append(wj); used.add(j)
            # 合併此 group 的 2D bbox & mask
            b = np.stack([g["bbox2d"] for g in group])
            x0, y0 = b[:,0].min(), b[:,1].min()
            x1      = (b[:,0]+b[:,2]).max()
            y1      = (b[:,1]+b[:,3]).max()
            bbox = (x0, y0, x1-x0, y1-y0)
            mask_union = np.zeros_like(group[0]["mask"])
            for g in group:
                mask_union |= g["mask"]
            cxm, cym = x0 + (x1-x0)/2, y0 + (y1-y0)/2
            merged.append({
                "category": "Walls",
                "bbox": bbox,
                "center": (cxm/W, cym/H),
                "mask": mask_union
            })

        doors_merged, comps_rest = merge_by_iou(raw_comps, 'Doors', W, H)
        windows_merged, comps_rest = merge_by_iou(comps_rest, 'Windows', W, H)       

        components = doors_merged + windows_merged + [
            {
                'category': c['category'],
                'bbox': tuple(c['bbox2d']),
                'center': c['center2d'],
                'mask': c['mask']
            }
            for c in comps_rest
        ]


    return rgb, sem, components

def build_scene_graph_from_components(components: List[Dict],
    W: int,
    H: int,
    depth: Optional[np.ndarray] = None,
    RELATION_RULES: Optional[Dict[Tuple[str,str], str]] = None,
    BBOX_PAD: float = 0.05, camera_intrinsics: Optional[Tuple[float, float, float, float]] = None,) -> nx.Graph:
    g = nx.Graph()
    pad = max(W, H) * BBOX_PAD

    fx = fy = cx = cy = None
    if camera_intrinsics is not None:
        fx, fy, cx, cy = camera_intrinsics
    # for idx, comp in enumerate(components):
    #     struct = comp['category']
    #     area = comp['mask'].sum() / (W * H)
    #     theta = 90 if comp['bbox'][3] > comp['bbox'][2] else 0
    #     g.add_node(idx,
    #                type=struct,
    #                cx=comp['center'][0], cy=comp['center'][1],
    #                w=comp['bbox'][2]/W, h=comp['bbox'][3]/H,
    #                area=area,
    #                depth=float(np.mean(depth[comp['mask']])) if depth is not None else 0.0,
    #                theta=theta,
    #                base=struct)

    for idx, comp in enumerate(components):
        # 2D 資訊保持不變
        cx2d, cy2d = comp['center']
        w2d, h2d   = comp['bbox'][2]/W, comp['bbox'][3]/H
        area2d     = comp['mask'].sum() / (W * H)

        w3d = h3d = d3d = vol3d = 0.0
        depth_std = 0.0
        center3d = [0.0, 0.0, 0.0]

        # --- 新增：3D bounding box & mask ---
        if depth is not None:
            ys, xs = np.where(comp['mask'])
            zs = depth[ys, xs]

            # 投影到相機座標
            Xs = (xs - cx) * zs / fx
            Ys = (ys - cy) * zs / fy
            pts3d = np.stack([Xs, Ys, zs], axis=1)

            min3d = pts3d.min(axis=0)
            max3d = pts3d.max(axis=0)
            # 3D 長寬高
            w3d, h3d, d3d = (max3d - min3d).tolist()
            # 3D “體素”數量 (proxy for 3D 面積/體積)
            vol3d   = float(pts3d.shape[0])
            depth_std = float(zs.std())
            center3d = pts3d.mean(axis=0).tolist()
            # 加到 node attributes
            node_attrs = {
                'cx': cx2d, 'cy': cy2d,
                'w2d': w2d, 'h2d': h2d, 'area2d': area2d,
                'w3d': float(w3d), 'h3d': float(h3d), 'area3d': vol3d,
                'depth_mean': float(zs.mean()), 'depth_std': depth_std,
                'center3d': center3d, 'base': comp['category'], 'type': comp['category'],
            }
        else:
            node_attrs = {
                'cx': cx2d, 'cy': cy2d,
                'w2d': w2d, 'h2d': h2d, 'area2d': area2d,
                # no 3d
                'w3d': 0, 'h3d': 0, 'area3d': 0,
                'center3d': 0,
                'depth_mean': 0, 'depth_std': 0,
                'base': comp['category'], 'type': comp['category'],
            }

        g.add_node(idx, **node_attrs)


    for i, j in itertools.combinations(g.nodes(), 2):
        bi, bj = g.nodes[i]['base'], g.nodes[j]['base']
        rel = RELATION_RULES.get(tuple(sorted([bi, bj])))
        if not rel:
            continue
        b1 = node_bbox(g, i, W, H, pad)
        b2 = node_bbox(g, j, W, H, pad)
        if compute_iou(b1, b2) == 0:
            continue
        dx = g.nodes[j]['cx'] - g.nodes[i]['cx']
        dy = g.nodes[j]['cy'] - g.nodes[i]['cy']
        g.add_edge(i, j,
                   relation=rel,
                   dx=float(dx), dy=float(dy),
                   dist=float(np.hypot(dx, dy)),
                   angle=float((np.degrees(np.arctan2(dy, dx)) + 360) % 360))
    return g

def graph_descriptor(g: nx.Graph, STRUCTURAL_CLASSES: List[str], EDGE_DIM: int, ANGLE_BINS: int,DIST_BINS: int) -> np.ndarray:

    if EDGE_DIM != ANGLE_BINS * DIST_BINS:
        raise ValueError(f"EDGE_DIM ({EDGE_DIM}) must equal ANGLE_BINS*DIST_BINS "
                         f"({ANGLE_BINS*DIST_BINS}).")

    if g.number_of_nodes() == 0:
        return np.zeros(len(STRUCTURAL_CLASSES) + len(STRUCTURAL_CLASSES) * 5 + EDGE_DIM, np.float32)
    
    hist = np.zeros(len(STRUCTURAL_CLASSES), np.float32)
    for _, d in g.nodes(data=True):
        if d['type'] in STRUCTURAL_CLASSES:
            hist[STRUCTURAL_CLASSES.index(d['type'])] += 1
        # attrs.append((d['w'], d['h'], d['area'], d['depth']))
    hist /= hist.sum() + 1e-6

    stats_per_class = []
    for cls in STRUCTURAL_CLASSES:
        cls_nodes = [d for _, d in g.nodes(data=True) if d['type'] == cls]
        if cls_nodes:  # 有對應節點
            w3d     = [d['w3d']        for d in cls_nodes]
            h3d     = [d['h3d']        for d in cls_nodes]
            area3d  = [d['area3d']     for d in cls_nodes]
            d_mean  = [d['depth_mean'] for d in cls_nodes]
            d_std   = [d['depth_std']  for d in cls_nodes]

            stats_per_class.extend([
                np.mean(w3d),
                np.mean(h3d),
                np.mean(area3d),
                np.mean(d_mean),
                np.mean(d_std),
            ])
        else:
            # 該類別在此圖中缺席 → 補 0
            stats_per_class.extend([0., 0., 0., 0., 0.])


    # stats = np.array([
    #     np.mean([a[0] for a in attrs]),
    #     np.mean([a[1] for a in attrs]),
    #     np.mean([a[2] for a in attrs]),
    #     np.std([a[3] for a in attrs]),
    #     g.number_of_nodes()/100
    # ], np.float32)


    edge_hist = np.zeros(EDGE_DIM, np.float32)
    if g.number_of_edges():
        vecs = np.array([[e.get('dx', 0.0), e.get('dy', 0.0)]
                         for _, _, e in g.edges(data=True)], np.float32)
        a, d_bins = polar_bins(vecs, ANGLE_BINS, DIST_BINS)
        mat = np.zeros((ANGLE_BINS, DIST_BINS), np.float32)
        valid = (a >= 0) & (a < ANGLE_BINS) & (d_bins >= 0) & (d_bins < DIST_BINS)
        a = a[valid]; d_bins = d_bins[valid]        
        
        for ai, di in zip(a, d_bins):
            mat[ai, di] += 1
        edge_hist = (mat / (mat.sum() + 1e-6)).flatten().astype(np.float32)
        
    return np.concatenate([hist, np.array(stats_per_class, np.float32), edge_hist]).astype(np.float32)

def graph_to_dict(g: nx.Graph, img_name: str, vec: np.ndarray) -> dict:
    nodes = []
    for n, d in g.nodes(data=True):
        nodes.append({
            'id': n,
            'category': d['type'],
            'center': d['center3d'],
            'w': d['w3d'], 'h': d['h3d'],
            'area': d['area3d'], 'depth': d['depth_std']        
            })
    edges = []
    for u, v, e in g.edges(data=True):
        edges.append({
            'from': u, 'to': v,
            'relation': e['relation'],
            'dist': e['dist'], 'angle': e['angle']
        })
    return {'image': img_name, 'nodes': nodes, 'edges': edges, 'descriptor': vec.tolist()}


def save_graph_and_seg(fn:str, rgb:np.ndarray, sem:np.ndarray, comps:list, g:nx.Graph, OUTPUT_VIS_FOLDER: Path):
    stem = Path(fn).stem.replace("_rgb_image", "")
    out_path = OUTPUT_VIS_FOLDER / f"{stem}_vis.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = rgb.shape[:2]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # 左：RGB + Graph
    axs[0].imshow(rgb)
    axs[0].axis("off")
    nodes = []
    for n, d in g.nodes(data=True):
        cx_px = d["cx"] * W
        cy_px = d["cy"] * H
        nodes.append((cx_px, cy_px, d["type"]))
    for cx, cy, cat in nodes:
        axs[0].text(cx, cy, cat,
                    fontsize=12, color="white",
                    bbox=dict(facecolor="black", alpha=0.5))
    for u, v, e in g.edges(data=True):
        c1 = (g.nodes[u]["cx"] * W, g.nodes[u]["cy"] * H)
        c2 = (g.nodes[v]["cx"] * W, g.nodes[v]["cy"] * H)
        axs[0].arrow(c1[0], c1[1], c2[0]-c1[0], c2[1]-c1[1],
                     color="cyan", head_width=10, alpha=0.5)

    # 右：Semantic + BBox
    axs[1].imshow(sem)
    axs[1].axis("off")
    for comp in comps:
        x, y, w, h = comp["bbox"]
        # draw box
        rect = Rectangle((x, y), w, h,
                         edgecolor="red", facecolor="none", linewidth=1)
        axs[1].add_patch(rect)
        # text at center
        cx_px = comp["center"][0] * W
        cy_px = comp["center"][1] * H
        axs[1].text(cx_px, cy_px, comp.get("category", ""),
                    fontsize=12, color="white",
                    bbox=dict(facecolor="black", alpha=0.5))

    plt.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)

def save_real_vis(fn: str, rgb: np.ndarray, comps: list, g: nx.Graph, OUTPUT_VIS_REAL: Path):
    stem = Path(fn).stem
    out_path = OUTPUT_VIS_REAL / f"{stem}_vis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = rgb.shape[:2]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # 左：RGB + scene graph overlay（不變）
    axs[0].imshow(rgb); axs[0].axis('off')
    for n, d in g.nodes(data=True):
        x_px = d['cx'] * W; y_px = d['cy'] * H
        axs[0].text(x_px, y_px, d['type'],
                    color='white', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.5))
    for u, v, e in g.edges(data=True):
        c1 = (g.nodes[u]['cx']*W, g.nodes[u]['cy']*H)
        c2 = (g.nodes[v]['cx']*W, g.nodes[v]['cy']*H)
        axs[0].arrow(c1[0], c1[1], c2[0]-c1[0], c2[1]-c1[1],
                     color='cyan', head_width=5, alpha=0.5)

    # 右：Mask + BBox（由原本的 RGB 改成 mask）
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for comp in comps:
        # 合併所有 comp 的 mask
        color = [random.randint(50, 255) for _ in range(3)]
        color_mask[comp['mask']] = color
    axs[1].imshow(color_mask)

    for comp in comps:
        x, y, w, h = comp['bbox']
        rect = Rectangle((x, y), w, h,
                         edgecolor='white', facecolor='none', linewidth=1)
        axs[1].add_patch(rect)
        cx_px = comp['center'][0] * W
        cy_px = comp['center'][1] * H
        axs[1].text(cx_px, cy_px, comp['category'],
                    color='white', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.5))
    plt.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    
    
    
    
