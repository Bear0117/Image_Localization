#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified PnP→ICP pipeline with safety gates.

- Base pose = top-1 synthetic image's pose (from --isaac_poses_csv).
- Stage-1 PnP (GlueStick) gives an initialization; if |t_pnp - t_base| > --pnp_max_jump_m → fall back to base.
- Stage-2 ICP (Open3D point-to-plane) refines; if |t_icp - t_init| > --icp_max_jump_m → fall back to init.
- Units are unified to meters internally. Output CSV uses meters and wxyz quaternion.

Requires:
- gluestick
- SuperGluePretrainedNetwork (for read_image)
- open3d
- opencv-python, numpy, torch, scipy

Author: merged for Bear (2025-08-08)
"""

import os, sys, json, csv, argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

# ---- GlueStick / SuperPoint pipeline ----
from gluestick import GLUESTICK_ROOT, batch_to_np, numpy_image_to_torch
from gluestick.models.two_view_pipeline import TwoViewPipeline
from SuperGluePretrainedNetwork.models.utils import read_image
from collections import defaultdict



# ==============================
# Shared math / pose utilities
# ==============================

def rotm_to_quat_wxyz(Rm: np.ndarray) -> Tuple[float, float, float, float]:
    q = Rotation.from_matrix(Rm).as_quat()  # x y z w
    return float(q[3]), float(q[0]), float(q[1]), float(q[2])

def quat_wxyz_to_rotm(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float64)

def se3_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def se3_distance_t(Ta: np.ndarray, Tb: np.ndarray) -> float:
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))

def scale_intrinsics(fx: float, fy: float, cx: float, cy: float,
                     base_w: int, base_h: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    if base_w <= 0 or base_h <= 0:
        return fx, fy, cx, cy
    sx = img_w / float(base_w)
    sy = img_h / float(base_h)
    return fx * sx, fy * sy, cx * sx, cy * sy

def valid_depth_mask(depth: np.ndarray, trunc: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0) & (depth <= trunc)

def depth_to_points_opencv(depth: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float,
                           stride: int, trunc: float) -> np.ndarray:
    h, w = depth.shape[:2]
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    d = depth[vv, uu].astype(np.float64)
    m = valid_depth_mask(d, trunc)
    uu = uu[m]; vv = vv[m]; d = d[m]
    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d
    return np.stack([X, Y, Z], axis=1)

def cv_to_usd(points_cv: np.ndarray) -> np.ndarray:
    if points_cv.size == 0:
        return points_cv
    T = np.diag([1.0, -1.0, -1.0])
    return (points_cv @ T.T)

def transform_points_h(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (pts_h @ T.T)[:, :3]
    return out


# ==============================
# I/O helpers
# ==============================

def read_pose_csv(path: str, translation_in_cm: bool = True) -> Dict[str, np.ndarray]:
    """Read pose CSV into dict: image_key -> T_world_cam (4x4, meters).
    Expected headers: [Camera Name, t_x, t_y, t_z, q_w, q_x, q_y, q_z]
    Keys are stored as '<Camera Name>_rgb_image.png' to match synth file names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pose CSV not found: {path}")
    pose_map: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            name = (r.get("Camera Name") or r.get("image") or r.get("filename") or "").strip()
            if not name:
                continue
            tx = float(r.get("t_x") or r.get("tx") or r.get("x") or 0.0)
            ty = float(r.get("t_y") or r.get("ty") or r.get("y") or 0.0)
            tz = float(r.get("t_z") or r.get("tz") or r.get("z") or 0.0)
            qw = float(r.get("q_w") or r.get("qw") or 1.0)
            qx = float(r.get("q_x") or r.get("qx") or 0.0)
            qy = float(r.get("q_y") or r.get("qy") or 0.0)
            qz = float(r.get("q_z") or r.get("qz") or 0.0)
            t = np.array([tx, ty, tz], dtype=np.float64)
            if translation_in_cm:
                t = t * 0.01  # cm -> m
            R = quat_wxyz_to_rotm(qw, qx, qy, qz)
            T = se3_from_rt(R, t)
            key = f"{name}_rgb_image.png"  # match synth filenames
            pose_map[key] = T
    return pose_map

def load_pairs_topk(path: str, topk: int = 10) -> List[Tuple[str, str]]:
    """Parse pairs JSON, return list of (real, synth_ref) for the first K refs per real.
       僅使用每個項目的 ref_image 欄位。"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    pairs: List[Tuple[str, str]] = []
    def push(real, synth):
        if real and synth:
            pairs.append((str(real), str(synth)))

    if isinstance(obj, dict) and "results" in obj:
        for rname, arr in obj["results"].items():
            if isinstance(arr, list) and len(arr) > 0:
                for ref in arr[:topk]:
                    if isinstance(ref, dict) and "ref_image" in ref:
                        push(rname, ref["ref_image"])
                    elif isinstance(ref, str):
                        push(rname, ref)  # 容錯：若直接是字串
    else:
        raise ValueError("Unsupported pairs JSON format for top-k.")

    if not pairs:
        raise ValueError("No pairs found in pairs JSON (top-k).")
    return pairs


# def load_pairs_top1(path: str) -> List[Tuple[str, str]]:
#     """Parse pairs JSON, return list of (real, synth_top1)."""
#     with open(path, "r", encoding="utf-8") as f:
#         obj = json.load(f)
#     pairs: List[Tuple[str, str]] = []
#     def push(real, synth):
#         if real and synth:
#             pairs.append((str(real), str(synth)))
#     if isinstance(obj, dict):
#         if "results" in obj:
#             for rname, arr in obj["results"].items():
#                 if isinstance(arr, list) and len(arr) > 0:
#                     ref = arr[0]
#                     if isinstance(ref, dict) and "ref_image" in ref:
#                         push(rname, ref["ref_image"])
#                     elif isinstance(ref, str):
#                         push(rname, ref)
#         elif "pairs" in obj and isinstance(obj["pairs"], list):
#             # [{"real_image": ..., "synth_image": ...}, ...]
#             for it in obj["pairs"]:
#                 if isinstance(it, dict):
#                     r = it.get("real") or it.get("real_image") or it.get("query")
#                     s = it.get("synthetic") or it.get("synth_image") or (it.get("top_k")[0] if isinstance(it.get("top_k"), list) and it.get("top_k") else None)
#                     push(r, s)
#         else:
#             # direct map {real: [refs]} or {real: synth}
#             for k, v in obj.items():
#                 if isinstance(v, list) and v:
#                     vv = v[0]
#                     push(k, vv["ref_image"] if isinstance(vv, dict) and "ref_image" in vv else vv)
#                 elif isinstance(v, str):
#                     push(k, v)
#     elif isinstance(obj, list):
#         for it in obj:
#             if isinstance(it, dict):
#                 r = it.get("real") or it.get("real_image") or it.get("query")
#                 s = it.get("synthetic") or it.get("synth_image")
#                 push(r, s)
#     if not pairs:
#         raise ValueError("No pairs found in pairs JSON (top-1).")
#     return pairs


# ==============================
# GlueStick pipeline + PnP (single pair, top-1 reference)
# ==============================

def create_gs_pipeline(dev: str) -> TwoViewPipeline:
    cfg = {
    "name": "two_view_pipeline",
    "use_lines": True,

    "extractor": {
        "name": "wireframe",

        # -------- SuperPoint 相關 --------
        "sp_params": {
            "force_num_keypoints": True,
            "max_num_keypoints": 2000,  # 調得越大越多點
        },

        # -------- Wireframe 相關 --------
        "wireframe_params": {
            "merge_points": True,
            "merge_line_endpoints": True,
            "nms_radius": 2,            # 2 比預設 3 更鬆
        },

        "max_n_lines": 400,             # 線段上限
    },

    "matcher": {
        "name": "gluestick",
        "weights": str(GLUESTICK_ROOT / "resources/weights/checkpoint_GlueStick_MD.tar"),
        "trainable": False,
    },

    "ground_truth": {"from_pose_depth": False},
    }
    return TwoViewPipeline(cfg).to(dev).eval()

def visualise_matches(img_u, img_k, kpts_u, kpts_k, matches, save_path: Path):
    h_u, w_u = img_u.shape[:2]
    h_k, w_k = img_k.shape[:2]
    canvas = np.zeros((max(h_u, h_k), w_u + w_k, 3), np.uint8)
    canvas[:h_u, :w_u] = img_u if img_u.ndim == 3 else cv2.cvtColor(img_u, cv2.COLOR_GRAY2BGR)
    canvas[:h_k, w_u:w_u + w_k] = img_k if img_k.ndim == 3 else cv2.cvtColor(img_k, cv2.COLOR_GRAY2BGR)
    for iu, ik in matches:
        x_u, y_u = kpts_u[iu]
        x_k, y_k = kpts_k[ik]
        cv2.line(canvas, (int(x_u), int(y_u)), (int(x_k) + w_u, int(y_k)), (0, 255, 0), 1)
    cv2.imwrite(str(save_path), canvas)

def depth_path_for(depth_root: Path, synth_rgb_name: str) -> Path:
    # Expected synth file key: <Camera Name>_rgb_image.png
    stem = Path(synth_rgb_name).stem.replace("_rgb_image", "")
    return depth_root / f"{stem}_distance_to_image_plane.npy"

def load_depth_any(p: Path) -> np.ndarray:
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"Failed to read depth: {p}")
    return d.astype(np.float32)

def triangulate_point(obs_uv: List[Tuple[float, float]], Ks: List[np.ndarray], poses: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    A = []
    for (u, v), K, (R, t) in zip(obs_uv, Ks, poses):
        P = K @ np.hstack([R, t.reshape(3, 1)])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.stack(A))
    X = Vt[-1][:3] / Vt[-1][3]
    return X.astype(np.float32)

def refine_point_LM(X0, obs, Ks, poses, it: int = 20):
    def res(X):
        X = X.reshape(3, 1)
        err = []
        for (u, v), K, (R, t) in zip(obs, Ks, poses):
            x_c = R.T @ X + (-R.T @ t).reshape(3, 1)
            if x_c[2] <= 0:
                err.extend([10, 10])
                continue
            px = K @ x_c
            err.extend([px[0, 0] / px[2, 0] - u, px[1, 0] / px[2, 0] - v])
        return np.array(err, np.float32)
    return least_squares(res, X0, method="lm", max_nfev=it).x.astype(np.float32)

def pnp_estimate_single(real_img_path: Path,
                        synth_img_path: Path,
                        synth_pose_map: Dict[str, np.ndarray],
                        depth_root: Path,
                        resize_hw: Tuple[int, int],
                        K_syn_ref: np.ndarray,
                        K_real_ref: np.ndarray,
                        device: str,
                        vis_out: Optional[Path]) -> Tuple[bool, Optional[np.ndarray], Dict]:
    """Estimate T_world_cam (PnP) for one (real, synth) pair using GlueStick matches and synth depth.
    Returns (ok, T_world_cam, diagnostics).
    """
    # Load images (GlueStick util handles resize & torch conversion)
    syn_img, _, _ = read_image(str(synth_img_path), device=device, resize=resize_hw, rotation=0, resize_float=resize_hw)
    real_img, _, _ = read_image(str(real_img_path),  device=device, resize=resize_hw, rotation=0, resize_float=resize_hw)
    syn_t = numpy_image_to_torch(syn_img)[None].to(device)
    real_t = numpy_image_to_torch(real_img)[None].to(device)

    # Depth for synth
    pose_key = os.path.basename(str(synth_img_path))
    dpth_path = depth_path_for(depth_root, pose_key)
    if not dpth_path.exists():
        return False, None, {"reason": f"depth_missing:{dpth_path.name}"}
    depth = cv2.resize(load_depth_any(dpth_path), resize_hw, interpolation=cv2.INTER_NEAREST)

    # Pose for synth (meters)
    if pose_key not in synth_pose_map:
        return False, None, {"reason": f"synth_pose_missing:{pose_key}"}
    T_world_cam_s = synth_pose_map[pose_key]
    R_ws = T_world_cam_s[:3, :3]; t_ws = T_world_cam_s[:3, 3]

    # Run GlueStick matching
    pipe = create_gs_pipeline(device)
    out = batch_to_np(pipe({"image0": syn_t, "image1": real_t}))
    k0, k1, m0 = out["keypoints0"], out["keypoints1"], out["matches0"]
    valid = m0 > -1
    matches = (np.stack([np.where(valid)[0], m0[valid]], 1)
               if valid.sum() > 0 else np.empty((0, 2), int))

    # --- Fundamental matrix RANSAC for image-level inlier ratio ---
    fm_inliers = 0
    fm_inlier_ratio = 0.0
    if matches.shape[0] >= 8:
        pts_syn  = k0[matches[:, 0]][:, :2].astype(np.float32)
        pts_real = k1[matches[:, 1]][:, :2].astype(np.float32)
        F, mask = cv2.findFundamentalMat(
            pts_syn, pts_real, cv2.FM_RANSAC, 1.0, 0.99
        )
        if F is not None and mask is not None:
            mask = mask.ravel().astype(bool)
            fm_inliers = int(mask.sum())
            if matches.shape[0] > 0:
                fm_inlier_ratio = fm_inliers / float(matches.shape[0])

    if vis_out is not None:
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        m_vis = matches[:, [1, 0]] if matches.shape[0] else matches
        visualise_matches(real_img, syn_img, k1, k0, m_vis, vis_out)

    if matches.shape[0] < 6:
        return False, None, {
            "reason": f"few_matches:{matches.shape[0]}",
            "matches": int(matches.shape[0]),
            "fm_inliers": int(fm_inliers),
            "fm_inlier_ratio": float(fm_inlier_ratio),
        }

    # Build 2D-3D correspondences via synth depth back-projection to WORLD
    h, w = depth.shape[:2]
    obj_pts = []
    img_pts = []
    for i_s, i_r in matches:
        u_s, v_s = k0[i_s]; u_r, v_r = k1[i_r]
        u_int, v_int = int(round(u_s)), int(round(v_s))
        if not (0 <= u_int < w and 0 <= v_int < h):
            continue
        d = float(depth[v_int, u_int])
        if not (np.isfinite(d) and d > 0):
            continue
        # synth camera coords
        Xc = np.linalg.inv(K_syn_ref) @ np.array([u_s, v_s, 1.0], np.float32)
        Xc *= d
        # world coords
        Xw = R_ws @ Xc + t_ws

        # Xc_cv  = np.linalg.inv(K_syn_ref) @ np.array([u_s, v_s, 1.0], np.float32)
        # Xc_cv *= d
        # Xc_usd = cv_to_usd(Xc_cv[np.newaxis, :])[0]
        # Xw     = R_ws @ Xc_usd + t_ws

        obj_pts.append(Xw)
        img_pts.append([u_r, v_r])

    if len(obj_pts) < 4:
        return False, None, {"reason": f"few_2d3d:{len(obj_pts)}"}

    obj_pts = np.asarray(obj_pts, np.float32)
    img_pts = np.asarray(img_pts, np.float32)

    # Solve PnP (world->camera pose as R_cw,t_cw)
    flags = cv2.SOLVEPNP_EPNP if len(obj_pts) < 6 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K_real_ref, None,
        iterationsCount=1000, reprojectionError=8.0, confidence=0.99, flags=flags
    )
    num_2d3d = len(obj_pts)
    if not ok or inliers is None or len(inliers) < 4:
        return False, None, {
            "reason": "pnp_fail",
            "matches": int(matches.shape[0]),
            "fm_inliers": int(fm_inliers),
            "fm_inlier_ratio": float(fm_inlier_ratio),
            "num_2d3d": int(num_2d3d),
        }
    inlier_idx = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_pts[inlier_idx], img_pts[inlier_idx],
        K_real_ref, None, rvec, tvec
    )

    # reprojection RMSE on PnP inliers
    proj, _ = cv2.projectPoints(
        obj_pts[inlier_idx], rvec, tvec, K_real_ref, None
    )
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts[inlier_idx], axis=1)
    pnp_reproj_rmse = float(np.sqrt(np.mean(err ** 2))) if err.size > 0 else float("nan")
    
    R_wc, _ = cv2.Rodrigues(rvec)
    R_cw = R_wc.T
    t_cw = -R_cw @ tvec.reshape(3)

    # Compose T_world_cam (meters)
    T_world_cam = se3_from_rt(R_cw, t_cw.astype(np.float64))
    diag = {
        "matches": int(matches.shape[0]),
        "fm_inliers": int(fm_inliers),
        "fm_inlier_ratio": float(fm_inlier_ratio),
        "num_2d3d": int(num_2d3d),
        "inliers": int(len(inliers)),
        "pnp_inlier_ratio": float(len(inliers) / float(num_2d3d)) if num_2d3d > 0 else float("nan"),
        "pnp_reproj_rmse": pnp_reproj_rmse,
    }
    return True, T_world_cam, diag


# ==============================
# ICP registration (Open3D)
# ==============================

def make_o3d_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    return pcd

def prep_pcd(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    return pcd

def icp_register_point_to_plane(src_pts_usd: np.ndarray, tgt_pts_world: np.ndarray,
                                T_init_world_cam: np.ndarray,
                                max_corr_list: List[float],
                                max_iter: int = 50,
                                voxel: float = 0.03) -> Tuple[np.ndarray, float, float]:
    # 不要預先 transform；讓 init 承擔初始對齊
    src_pcd_0 = prep_pcd(make_o3d_pcd(src_pts_usd), voxel)
    tgt_pcd   = prep_pcd(make_o3d_pcd(tgt_pts_world), voxel)

    T = T_init_world_cam.copy()
    reg = None
    for dist in max_corr_list:
        reg = o3d.pipelines.registration.registration_icp(
            src_pcd_0, tgt_pcd, max_correspondence_distance=dist, init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        T = reg.transformation  # 只更新 init，source 幾何保持不變
    if reg is None:
        return T_init_world_cam.copy(), 0.0, 1e9
    return reg.transformation, float(reg.fitness), float(reg.inlier_rmse)


# ==============================
# Simple offscreen rendering to PNG (optional)
# ==============================

def render_pcds_png(pcd_list: List[o3d.geometry.PointCloud], save_path: Path):
    try:
        from open3d.visualization import rendering
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(np.concatenate([np.asarray(p.points) for p in pcd_list], axis=0))
        )
        center = bbox.get_center()
        extents = bbox.get_extent()
        radius = float(np.linalg.norm(extents)) * 1.2 + 1e-3
        renderer = rendering.OffscreenRenderer(1280, 960)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])
        for i, p in enumerate(pcd_list):
            scene.add_geometry(f"pc{i}", p, mat)
        cam = scene.camera
        eye = center + np.array([radius, radius, radius])
        up = np.array([0, 1, 0])
        scene.camera.look_at(center, eye, up)
        img = renderer.render_to_image()
        o3d.io.write_image(str(save_path), img)
        renderer.release()
    except Exception as e:
        # If offscreen rendering is not available, silently skip
        pass

def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    R_rel = Rb @ Ra.T
    tr = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(tr)))
    return ang

def pnp_estimate_multi(real_img_path: Path,
                       synth_img_paths: List[Path],
                       synth_pose_map: Dict[str, np.ndarray],
                       depth_root: Path,
                       resize_hw: Tuple[int, int],
                       K_syn_ref: np.ndarray,
                       K_real_ref: np.ndarray,
                       device: str,
                       vis_dir: Optional[Path],
                       junc_points_fullres: Optional[np.ndarray],
                       junc_tol_px: float) -> Tuple[bool, Optional[np.ndarray], Dict]:
    # 讀 real 圖（一次）
    real_img, _, _ = read_image(str(real_img_path), device=device, resize=resize_hw, rotation=0, resize_float=resize_hw)
    real_t = numpy_image_to_torch(real_img)[None].to(device)

    # 將（可選）junction 依 resize 比例縮放到工作解析度
    juncs_scaled = None
    if junc_points_fullres is not None and junc_points_fullres.size:
        w0 = float(np.max(junc_points_fullres[:, 0]) + 1e-5)
        h0 = float(np.max(junc_points_fullres[:, 1]) + 1e-5)
        sx = resize_hw[0] / w0
        sy = resize_hw[1] / h0
        juncs_scaled = junc_points_fullres * np.array([sx, sy], np.float32)

    pipe = create_gs_pipeline(device)

    # 聚合：以 (rounded u_r, v_r) 當 key，收集多參考的觀測
    match_groups = defaultdict(list)
    total_kept = 0
    total_matches = 0

    if vis_dir is not None:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    for sp in synth_img_paths:
        pose_key = os.path.basename(str(sp))
        if pose_key not in synth_pose_map:
            continue
        # 讀合成影像與對應深度
        syn_img, _, _ = read_image(str(sp), device=device, resize=resize_hw, rotation=0, resize_float=resize_hw)
        syn_t = numpy_image_to_torch(syn_img)[None].to(device)
        dpth_path = depth_path_for(depth_root, pose_key)
        if not dpth_path.exists():
            continue
        depth = cv2.resize(load_depth_any(dpth_path), resize_hw, interpolation=cv2.INTER_NEAREST)

        out = batch_to_np(pipe({"image0": syn_t, "image1": real_t}))
        k0, k1, m0 = out["keypoints0"], out["keypoints1"], out["matches0"]
        valid = m0 > -1
        matches = (np.stack([np.where(valid)[0], m0[valid]], 1)
                   if valid.sum() > 0 else np.empty((0, 2), int))

        # G1：每張參考各自輸出可視覺化
        if vis_dir is not None:
            m_vis = matches[:, [1, 0]] if matches.shape[0] else matches
            visualise_matches(real_img, syn_img, k1, k0, m_vis,
                              Path(vis_dir) / f"vis_{Path(sp).stem}.png")

        if matches.shape[0] == 0:
            continue

        total_matches += matches.shape[0]

        # 將深度反投到世界座標
        T_ws = synth_pose_map[pose_key]
        R_ws = T_ws[:3, :3]; t_ws = T_ws[:3, 3]
        h, w = depth.shape[:2]

        kept_pairs = []
        for i_s, i_r in matches:
            u_s, v_s = k0[i_s]; u_r, v_r = k1[i_r]
            u_int, v_int = int(round(u_s)), int(round(v_s))
            if not (0 <= u_int < w and 0 <= v_int < h):
                continue
            d = float(depth[v_int, u_int])
            if not (np.isfinite(d) and d > 0):
                continue

            # （可選）junction 篩選：僅保留靠近結構點的匹配
            if juncs_scaled is not None:
                dv = juncs_scaled - np.array([u_r, v_r], np.float32)
                if (dv[:, 0] ** 2 + dv[:, 1] ** 2).min() > (junc_tol_px ** 2):
                    continue

            Xc = np.linalg.inv(K_syn_ref) @ np.array([u_s, v_s, 1.0], np.float32)
            Xc *= d
            Xw = R_ws @ Xc + t_ws
            # Xc_cv  = np.linalg.inv(K_syn_ref) @ np.array([u_s, v_s, 1.0], np.float32)
            # Xc_cv *= d
            # Xc_usd = cv_to_usd(Xc_cv[np.newaxis, :])[0]
            # Xw     = R_ws @ Xc_usd + t_ws

            kept_pairs.append((i_s, i_r))
            match_groups[(int(round(u_r)), int(round(v_r)))].append((Xw, K_syn_ref, (R_ws, t_ws)))

        total_kept += len(kept_pairs)

    if total_kept < 6 or len(match_groups) == 0:
        return False, None, {
            "reason": f"few_multi_matches:{total_kept}",
            "matches": int(total_matches),
            "kept_matches": int(total_kept),
            "groups": int(len(match_groups)),
        }

    # 對每個 real 像素群做 3D 決策：>=3 觀測 → 三角化+LM；否則取第一個
    uv_r, Xws = [], []
    for (u, v), obs in match_groups.items():
        if len(obs) >= 3:
            obs_uv, Ks, poses = [], [], []
            for Xw, K_s, (R_s, t_s) in obs:
                R_cw = R_s.T; t_cw = -R_cw @ t_s
                x_c = R_cw @ Xw + t_cw
                px = K_s @ x_c
                obs_uv.append((px[0]/px[2], px[1]/px[2]))
                Ks.append(K_s); poses.append((R_s, t_s))
            X0 = triangulate_point(obs_uv, Ks, poses)
            X_f = refine_point_LM(X0, obs_uv, Ks, poses, it=20)
        else:
            X_f = obs[0][0]
        uv_r.append([u, v]); Xws.append(X_f)

    obj_pts = np.array(Xws, np.float32)
    img_pts = np.array(uv_r, np.float32)
    if len(obj_pts) < 4:
        return False, None, {"reason": f"few_2d3d_after_group:{len(obj_pts)}"}

    N = min(len(obj_pts), len(img_pts))
    obj_pts, img_pts = obj_pts[:N], img_pts[:N]
    flags = cv2.SOLVEPNP_EPNP if N < 6 else cv2.SOLVEPNP_ITERATIVE

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K_real_ref, None,
        iterationsCount=1000, reprojectionError=8.0, confidence=0.99, flags=flags
    )

    num_2d3d = len(obj_pts)
    if not ok or inliers is None or len(inliers) < 4:
        return False, None, {
            "reason": "pnp_ransac_fail",
            "matches": int(total_matches),
            "kept_matches": int(total_kept),
            "groups": int(len(match_groups)),
            "num_2d3d": int(num_2d3d),
        }

    inlier_idx = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_pts[inlier_idx], img_pts[inlier_idx],
        K_real_ref, None, rvec, tvec
    )
    proj, _ = cv2.projectPoints(
        obj_pts[inlier_idx], rvec, tvec, K_real_ref, None
    )
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts[inlier_idx], axis=1)
    pnp_reproj_rmse = float(np.sqrt(np.mean(err ** 2))) if err.size > 0 else float("nan")

    R_wc, _ = cv2.Rodrigues(rvec)
    R_cw = R_wc.T
    t_cw = -R_cw @ tvec.reshape(3)

    T_world_cam = se3_from_rt(R_cw, t_cw.astype(np.float64))
    diag = {
        "matches": int(total_matches),
        "kept_matches": int(total_kept),
        "groups": int(len(match_groups)),
        "num_2d3d": int(num_2d3d),
        "inliers": int(len(inliers)),
        "pnp_inlier_ratio": float(len(inliers) / float(num_2d3d)) if num_2d3d > 0 else float("nan"),
        "pnp_reproj_rmse": pnp_reproj_rmse,
        "fm_inliers": 0,
        "fm_inlier_ratio": float("nan"),
    }
    return True, T_world_cam, diag



def global_register_fpfh(src_pts_usd: np.ndarray, tgt_pts_world: np.ndarray,
                         T_init_world_cam: np.ndarray,
                         max_corr_list: List[float],
                         max_iter: int = 50,
                         voxel: float = 0.03) -> Tuple[np.ndarray, float, float]:
    src_pcd = prep_pcd(make_o3d_pcd(src_pts_usd), voxel)
    tgt_pcd = prep_pcd(make_o3d_pcd(tgt_pts_world), voxel)
    
    # 計算FPFH特徵（voxel*5作為半徑，確保特徵穩定）
    radius_feature = voxel * 10
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # 全局配準（使用RANSAC，忽略T_init或設為identity）
    checker = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
    checker_dist = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr_list[0])  # 使用您的max_corr_list第一個作為初始閾值
    reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd, tgt_pcd, src_fpfh, tgt_fpfh, mutual_filter=False,
        max_correspondence_distance=max_corr_list[0],
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, checkers=[checker, checker_dist],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 0.999)
    )
    
    # 可選：用ICP細化（point-to-plane，如原代碼）
    if reg.fitness > 0.1:  # 只在全局成功時細化
        reg = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, max_correspondence_distance=max_corr_list[-1], init=reg.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
    
    return reg.transformation, float(reg.fitness), float(reg.inlier_rmse)


def make_camera_frame(T_world_cam: np.ndarray, size: float = 0.8) -> o3d.geometry.TriangleMesh:
    """Create a colored coordinate frame at camera pose."""
    fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    fr.transform(T_world_cam.astype(np.float64))
    return fr

def render_geoms_png(geoms: List[o3d.geometry.Geometry], save_path: Path):
    """Render any O3D geometries (pcd / mesh / lines) to PNG offscreen."""
    try:
        from open3d.visualization import rendering
        # union AABB for all geometries
        bbox_min = None; bbox_max = None
        for g in geoms:
            b = g.get_axis_aligned_bounding_box()
            mn = b.get_min_bound(); mx = b.get_max_bound()
            if bbox_min is None:
                bbox_min, bbox_max = mn, mx
            else:
                bbox_min = np.minimum(bbox_min, mn)
                bbox_max = np.maximum(bbox_max, mx)
        bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
        center = bbox.get_center()
        extents = bbox.get_extent()
        radius = float(np.linalg.norm(extents)) * 1.2 + 1e-3

        renderer = rendering.OffscreenRenderer(1280, 960)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])

        for i, g in enumerate(geoms):
            scene.add_geometry(f"g{i}", g, mat)

        eye = center + np.array([radius, radius, radius])
        up = np.array([0, 1, 0])
        scene.camera.look_at(center, eye, up)
        img = renderer.render_to_image()
        o3d.io.write_image(str(save_path), img)
        renderer.release()
    except Exception:
        pass


# ==============================
# Main orchestration
# ==============================

def main():
    ap = argparse.ArgumentParser(description="Unified PnP→ICP localization with safety gates (top-1 pair).")
    ap.add_argument("--pairs_json", required=True, help="Pairs JSON; only the first synthetic per real is used.")
    ap.add_argument("--real_rgb_dir", required=True)
    ap.add_argument("--real_depth_dir", required=True, help="Real depth dir (npy or png); meters.")
    ap.add_argument("--synth_rgb_dir", required=True)
    ap.add_argument("--synth_depth_dir", required=True, help="Synthetic depth dir; used for back-projection.")
    ap.add_argument("--isaac_poses_csv", required=True, help="Synthetic camera poses CSV (cm, wxyz).")
    ap.add_argument("--real_gt_csv", required=False, help="Optional real GT poses CSV (cm, wxyz) for evaluation.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pnp_max_jump_m", type=float, default=4.0, help="If |t_pnp - t_base| > threshold, fall back to base for ICP init.")
    ap.add_argument("--icp_max_jump_m", type=float, default=4.0, help="If |t_icp - t_init| > threshold, keep init as final.")
    ap.add_argument("--min_fitness", type=float, default=0.15)
    ap.add_argument("--max_rmse", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1, help="Depth subsample stride.")
    ap.add_argument("--depth_trunc", type=float, default=40.0)
    ap.add_argument("--voxel_size", type=float, default=0.01)
    ap.add_argument("--vis", action="store_true", help="Save PNG visualizations for PnP matches and ICP before/after.")
    ap.add_argument("--neighbor_rot_deg", type=float, default=90.0,help="鄰近影像旋轉差閾值 (deg)")
    ap.add_argument("--neighbor_dist_m", type=float, default=3.0,help="鄰近影像位置差閾值 (m)")
    ap.add_argument("--neighbor_name_prefixes", nargs="+", default=None,help="僅納入檔名開頭符合任一前綴的合成影像（可選）")
    ap.add_argument("--junction_json", type=str, default=None,help="結構點 JSON（可選，用於在 real 圖上做像素級篩選）")
    ap.add_argument("--junction_tol_px", type=float, default=10.0,help="junction 像素容許誤差半徑")
    ap.add_argument("--topk_refs", type=int, default=1,help="對每個 real 使用前 K 個 ref_image（依 JSON 列表順序）")
    ap.add_argument("--save_ply", action="store_true",help="Save pre-ICP point clouds (OpenCV camera coords) to ./output_ply_beforeICP/")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_ply_dir = Path("output_ply_beforeICP")
    if args.save_ply:
        out_ply_dir.mkdir(parents=True, exist_ok=True)
    est_csv = os.path.join(args.out_dir, "final_poses.csv")
    log_csv = os.path.join(args.out_dir, "debug_log.csv")

    # Load pairs (top-1 only)
    # pairs = load_pairs_top1(args.pairs_json)
    pairs = load_pairs_topk(args.pairs_json, topk=args.topk_refs)


    # Load synth poses (meters)
    synth_pose_map = read_pose_csv(args.isaac_poses_csv, translation_in_cm=True)

    # Optional real GT
    junction_map = {}
    if args.junction_json and os.path.exists(args.junction_json):
        with open(args.junction_json, "r", encoding="utf-8") as f:
            junction_map = json.load(f)  # { "<real_image_name>": [[x,y], ...], ... }

    real_gt_map = {}
    if args.real_gt_csv and os.path.exists(args.real_gt_csv):
        real_gt_map = {}
        with open(args.real_gt_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                name = (r.get("Camera Name") or r.get("image") or "").strip()
                if not name: 
                    continue
                tx = float(r.get("t_x") or r.get("tx") or 0.0) * 0.01
                ty = float(r.get("t_y") or r.get("ty") or 0.0) * 0.01
                tz = float(r.get("t_z") or r.get("tz") or 0.0) * 0.01
                qw = float(r.get("q_w") or r.get("qw") or 1.0)
                qx = float(r.get("q_x") or r.get("qx") or 0.0)
                qy = float(r.get("q_y") or r.get("qy") or 0.0)
                qz = float(r.get("q_z") or r.get("qz") or 0.0)
                T = se3_from_rt(quat_wxyz_to_rotm(qw, qx, qy, qz), np.array([tx, ty, tz], dtype=np.float64))
                # keys accepted: full path, basename, stem
                base = os.path.basename(name)
                stem = os.path.splitext(base)[0]
                real_gt_map[name] = T
                real_gt_map[base] = T
                real_gt_map[stem] = T

    # GlueStick intrinsics (scaled to resize)
    resize = (720, 540)  # (w,h)
    K_syn = np.array([[2198.46 * resize[0] / 1920, 0, 960 * resize[0] / 1920],
                      [0, 2198.46 * resize[1] / 1080, 540 * resize[1] / 1080],
                      [0, 0, 1]], np.float32)
    K_real = np.array([[1650 * resize[0] / 2400, 0, 1200 * resize[0] / 2400],
                       [0, 1650 * resize[1] / 1347, 673.5 * resize[1] / 1347],
                       [0, 0, 1]], np.float32)

    # Per-pair loop
    results = []
    with open(est_csv, "w", newline="", encoding="utf-8") as f_est, \
         open(log_csv, "w", newline="", encoding="utf-8") as f_log:
        w_est = csv.writer(f_est)
        w_log = csv.writer(f_log)
        
        w_est.writerow([
            "real_image", "synth_image",
            "tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz",
            "fitness", "rmse",
            "delta_pnp_m", "delta_icp_m",
            "rot_pnp_deg", "rot_icp_deg",
            "pnp_matches", "pnp_num_2d3d", "pnp_inliers",
            "pnp_inlier_ratio", "pnp_reproj_rmse",
            "fm_inliers", "fm_inlier_ratio",
            "init_source", "refined", "fallback",
            "trans_err_m", "rot_err_deg"
        ])
        w_log.writerow(["real_image", "pnp_matches", "pnp_inliers", "fitness", "rmse"])

        for idx, (real_name, synth_name) in enumerate(pairs):
            # real_path = Path(args.real_rgb_dir) / (os.path.basename(real_name) if not os.path.isabs(real_name) else os.path.basename(real_name))
            # synth_path = Path(args.synth_rgb_dir) / (os.path.basename(synth_name) if not os.path.isabs(synth_name) else os.path.basename(synth_name))
            real_path  = Path(real_name)  if os.path.isabs(real_name)  else Path(args.real_rgb_dir)  / real_name
            synth_path = Path(synth_name) if os.path.isabs(synth_name) else Path(args.synth_rgb_dir) / synth_name

            # Base pose from top-1 synthetic
            synth_key = os.path.basename(str(synth_path))
            if synth_key not in synth_pose_map:
                print(f"[WARN] No synthetic pose for {synth_key}; skip")
                continue
            T_base = synth_pose_map[synth_key]

            # ── 以 Top-1 為基準擴充鄰近參考 ───────────────────────────────────────────
            ref_list = [synth_path]  # 第一張就是 pairs_json 的 Top-1
            base_key = os.path.basename(str(synth_path))
            R0 = T_base[:3, :3]; t0 = T_base[:3, 3]
            existing = {base_key}

            for key, T in synth_pose_map.items():
                if key in existing or key == base_key:
                    continue
                # （可選）名稱前綴過濾
                if args.neighbor_name_prefixes:
                    stem = Path(key).stem
                    if not any(stem.startswith(p) for p in args.neighbor_name_prefixes):
                        continue
                Rk, tk = T[:3, :3], T[:3, 3]
                ang = rotation_error_deg(R0, Rk)       # 已有函式
                dist = float(np.linalg.norm(tk - t0))
                if ang < args.neighbor_rot_deg and dist < args.neighbor_dist_m:
                    ref_list.append(Path(args.synth_rgb_dir) / key)
                    existing.add(key)

            # Stage-1 PnP
            pnp_png = Path(args.out_dir) / f"{Path(real_name).stem}_pnp_matches.png" if args.vis else None
            if len(ref_list) > 1:
                junc = None
                if junction_map:
                    # 支援用 full-res junction（若提供）提升幾何穩定度
                    junc = np.array(junction_map.get(os.path.basename(str(real_path)), []), np.float32)
                ok_pnp, T_pnp, diag = pnp_estimate_multi(
                    real_path, ref_list, synth_pose_map, Path(args.synth_depth_dir),
                    resize, K_syn, K_real, device="cuda" if torch.cuda.is_available() else "cpu",
                    vis_dir=(Path(args.out_dir) / f"{Path(real_name).stem}_multi_vis") if args.vis else None,
                    junc_points_fullres=junc, junc_tol_px=args.junction_tol_px
                )
            else:
                ok_pnp, T_pnp, diag = pnp_estimate_single(
                    real_path, synth_path, synth_pose_map, Path(args.synth_depth_dir),
                    resize, K_syn, K_real, "cuda" if torch.cuda.is_available() else "cpu", pnp_png
                )

                        # Collect PnP diagnostics
            if isinstance(diag, dict):
                pnp_matches      = int(diag.get("matches", 0))
                pnp_num_2d3d     = int(diag.get("num_2d3d", 0))
                pnp_inliers      = int(diag.get("inliers", 0))
                pnp_inlier_ratio = float(diag.get("pnp_inlier_ratio", float("nan")))
                pnp_reproj_rmse  = float(diag.get("pnp_reproj_rmse", float("nan")))
                fm_inliers       = int(diag.get("fm_inliers", 0))
                fm_inlier_ratio  = float(diag.get("fm_inlier_ratio", float("nan")))
            else:
                pnp_matches = pnp_num_2d3d = pnp_inliers = fm_inliers = 0
                pnp_inlier_ratio = pnp_reproj_rmse = fm_inlier_ratio = float("nan")

            

            init_source = "base"
            delta_pnp = None
            rot_pnp_deg = float("nan")
            T_init = T_base.copy()
            if ok_pnp and T_pnp is not None:
                delta_pnp = se3_distance_t(T_pnp, T_base)
                rot_pnp_deg = rotation_error_deg(T_pnp[:3, :3], T_base[:3, :3])
                if delta_pnp <= args.pnp_max_jump_m:
                    T_init = T_pnp
                    init_source = "pnp"
                else:
                    init_source = "base"
            else:
                init_source = "base"

            # Stage-2 ICP
            # Real depth
            # Expect real depth file: <real_name>.npy or .png under real_depth_dir (meters)
            rp_base = os.path.basename(str(real_path))
            rp_stem = os.path.splitext(rp_base)[0]
            # Try several filename patterns
            cand = [Path(args.real_depth_dir) / f"{rp_stem}.npy",
                    Path(args.real_depth_dir) / f"{rp_base}.npy",
                    Path(args.real_depth_dir) / f"{rp_stem}.png",
                    Path(args.real_depth_dir) / f"{rp_base}.png"]
            real_depth_p = None
            for c in cand:
                if c.exists():
                    real_depth_p = c; break
            if real_depth_p is None:
                print(f"[WARN] Missing real depth for {rp_base}; skip ICP")
                continue
            # Real depth
            d_real = load_depth_any(real_depth_p)
            H_r, W_r = d_real.shape[:2]
            frx, fry, frcx, frcy = scale_intrinsics(
                float(K_real[0,0]), float(K_real[1,1]), float(K_real[0,2]), float(K_real[1,2]),
                base_w=720, base_h=540, img_w=W_r, img_h=H_r  # 這裡 base_w,h 要放 K_real 對應的基準尺寸
            )
            real_pts_cv = depth_to_points_opencv(d_real, frx, fry, frcx, frcy, stride=args.stride, trunc=args.depth_trunc)
            # to correct units.
            # real_pts_cv *= 5.0
            real_pts_usd = cv_to_usd(real_pts_cv)

            # Synth depth
            d_synth = load_depth_any(depth_path_for(Path(args.synth_depth_dir), synth_key))
            H_s, W_s = d_synth.shape[:2]
            fsx, fsy, fscx, fscy = scale_intrinsics(
                float(K_syn[0,0]), float(K_syn[1,1]), float(K_syn[0,2]), float(K_syn[1,2]),
                base_w=720, base_h=540, img_w=W_s, img_h=H_s  # 同上，基於 K_syn 的基準
            )
            synth_pts_cv = depth_to_points_opencv(d_synth, fsx, fsy, fscx, fscy, stride=args.stride, trunc=args.depth_trunc)
            # to correct units.
            # synth_pts_cv *= 5.0
            synth_pts_usd = cv_to_usd(synth_pts_cv)
            synth_pts_world = transform_points_h(T_base, synth_pts_usd)



            # ---- Save pre-ICP point clouds in OpenCV camera coordinates (no world/USB transform) ----
            if args.save_ply:
                # real: OpenCV camera coords
                if real_pts_cv.size > 0:
                    # real_pts_cv *= 3.281  # to feet
                    pcd_real_cv = make_o3d_pcd(real_pts_cv)  # 直接用 CV 座標，勿轉換
                    o3d.io.write_point_cloud(str(out_ply_dir / f"{rp_stem}__real_cv.ply"), pcd_real_cv)

                # synth (Top-1): OpenCV camera coords
                synth_stem = os.path.splitext(os.path.basename(str(synth_path)))[0]
                if synth_pts_cv.size > 0:
                    # synth_pts_cv *= 3.281  # to feet
                    pcd_synth_cv = make_o3d_pcd(synth_pts_cv)  # 直接用 CV 座標，勿轉換
                    o3d.io.write_point_cloud(str(out_ply_dir / f"{rp_stem}__{synth_stem}__synth_cv.ply"), pcd_synth_cv)



            # ICP multi-stage
            # T_icp, fitness, rmse = icp_register_point_to_plane(
            #     real_pts_usd, synth_pts_world, T_init, max_corr_list=[0.30, 0.20, 0.10, 0.05], max_iter=50, voxel=args.voxel_size
            # )
            T_icp, fitness, rmse = global_register_fpfh(
                real_pts_usd, synth_pts_world, T_init, max_corr_list=[0.30, 0.20, 0.10, 0.05], max_iter=50, voxel=args.voxel_size
            )
            delta_icp = se3_distance_t(T_icp, T_init)
            rot_icp_deg = rotation_error_deg(T_icp[:3, :3], T_init[:3, :3])

            pass_quality = (delta_icp <= args.icp_max_jump_m) and (fitness >= args.min_fitness) and (rmse <= args.max_rmse)
            refined = bool(pass_quality)
            fallback = ""
            T_final = T_icp if pass_quality else T_init
            if not pass_quality:
                fallback = f"icp_guard(delta={delta_icp:.2f}, fitness={fitness:.3f}, rmse={rmse:.3f})"

            # Visualization
            if args.vis:
                # before ICP
                src_init_pts = transform_points_h(T_init, real_pts_usd)
                src_init = make_o3d_pcd(src_init_pts);  src_init.paint_uniform_color([0.60, 0.80, 1.00])  # 淺藍
                tgt = make_o3d_pcd(synth_pts_world);    tgt.paint_uniform_color([0.70, 0.90, 0.70])        # 淺綠
                render_geoms_png([src_init, tgt], Path(args.out_dir) / f"{rp_stem}_icp_before.png")

                # after ICP (or fallback to init if guard fails)
                src_final_pts = transform_points_h(T_final, real_pts_usd)
                src_final = make_o3d_pcd(src_final_pts); src_final.paint_uniform_color([0.60, 0.80, 1.00]) # 淺藍
                render_geoms_png([src_final, tgt], Path(args.out_dir) / f"{rp_stem}_icp_after.png")

            # Optionally compute errors vs GT
            # Optionally compute errors vs GT
            trans_err_m = float("nan")
            rot_err_deg = float("nan")
            if 'real_gt_map' in locals() and real_gt_map:
                gt_T = None
                rn = str(real_name)
                candidates = [rn, os.path.basename(rn), os.path.splitext(os.path.basename(rn))[0]]
                for k in candidates:
                    if k in real_gt_map:
                        gt_T = real_gt_map[k]
                        break
                if gt_T is not None:
                    trans_err_m = se3_distance_t(T_final, gt_T)
                    rot_err_deg = rotation_error_deg(T_final[:3, :3], gt_T[:3, :3])

            tx, ty, tz = T_final[:3, 3]
            qw, qx, qy, qz = rotm_to_quat_wxyz(T_final[:3, :3])


            w_est.writerow([
                os.path.basename(real_name),
                os.path.basename(synth_name),
                f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}",
                f"{qw:.8f}", f"{qx:.8f}", f"{qy:.8f}", f"{qz:.8f}",
                f"{fitness:.6f}", f"{rmse:.6f}",
                f"{(delta_pnp if delta_pnp is not None else np.nan):.6f}",
                f"{delta_icp:.6f}",
                f"{rot_pnp_deg:.6f}", f"{rot_icp_deg:.6f}",
                pnp_matches, pnp_num_2d3d, pnp_inliers,
                f"{pnp_inlier_ratio:.6f}",
                f"{pnp_reproj_rmse:.6f}",
                fm_inliers, f"{fm_inlier_ratio:.6f}",
                init_source, int(refined), fallback,
                f"{trans_err_m:.6f}", f"{rot_err_deg:.6f}"
            ])
            w_log.writerow([os.path.basename(real_name), diag.get("matches", 0) if isinstance(diag, dict) else 0,
                            diag.get("inliers", 0) if isinstance(diag, dict) else 0, f"{fitness:.6f}", f"{rmse:.6f}"])

            print(f"[{idx+1}/{len(pairs)}] {os.path.basename(real_name)} | init={init_source} | "
                  f"fitness={fitness:.3f}, rmse={rmse:.3f}, Δpnp={delta_pnp if delta_pnp is not None else np.nan:.2f} m, "
                  f"Δicp={delta_icp:.2f} m")

    print(f"[OK] Saved poses -> {est_csv}")
    print(f"[OK] Saved logs  -> {log_csv}")


if __name__ == "__main__":
    main()
