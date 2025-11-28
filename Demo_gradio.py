import os
import json
import shutil
import subprocess
import uuid
from pathlib import Path
import inference

import gradio as gr
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


import torch
from gluestick import GLUESTICK_ROOT, batch_to_np, numpy_image_to_torch
from gluestick.models.two_view_pipeline import TwoViewPipeline


# === 你自己的 module，請確認檔名 ===
from construct_real_graph_1127 import build_real_graph_single_image
import graph_bear_Demo_1127 as retr  # 裡面有 main() / _resolve_img_path

# ---------------------------------------------------------
# 0. 一些固定路徑（請依你實際環境修改）
# ---------------------------------------------------------

# 暫存資料夾（每次呼叫建立一個 UUID 子資料夾）
TMP_ROOT = Path("./tmp_demo")
TMP_ROOT.mkdir(exist_ok=True, parents=True)

# 候選 graph / wireframe / 影像的路徑
CAND_GRAPH_JSON = Path(r"C:\Users\User\Delta_Dataset\ReadingRoom\Synth\Graph\synthetic_graphs_1121.json")      # synth graph JSON
CAND_IMAGE_ROOT = Path(r"C:\Users\User\Delta_Dataset\ReadingRoom\Synth\RGB")       # synth RGB root
WIRE_CAND_JSON  = Path(r"C:\Users\User\Delta_Dataset\ReadingRoom\Synth\Wireframe\inference_results_synth.json")        # wireframe DB for candidates

# HAWP / wireframe inference
HAWP_CONFIG = Path(r"C:\Users\User\Image_Localization\data\plnet.yaml")
HAWP_CKPT   = Path(r"C:\Users\User\Delta_Dataset\plnet.pth")
# INFERENCE_PY = Path(r"C:\Users\User\Image_Localization\inference_1127.py")  # 你剛剛改過的 inference.py

# --- Fine localization 相關的固定路徑（請依實際環境修改） ---

# localize_pnp_icp_index.py 的路徑
LOCALIZE_PNP_ICP_SCRIPT = Path(r"C:\Users\User\Image_Localization\localize_pnp_icp_index.py")
REAL_DEPTH_DIR = Path(r"C:\Users\User\Image_Localization\tmp\depth")
SYNTH_DEPTH_DIR = Path(r"C:\Users\User\Delta_Dataset\ReadingRoom\Synth\Depth")

# Isaac SIM 相機 pose 的 CSV（cm + wxyz）
ISAAC_POSES_CSV = Path(r"C:\Users\User\Delta_Dataset\ReadingRoom\ReadingRoom_poses.csv")

# 房間 mesh（建議 PLY）
ROOM_MESH_PATH = Path(r"C:\Users\User\Delta_Dataset\reading_room_Delta_obj.ply")


# ---------------------------------------------------------
# 1. Wireframe 視覺化：把線畫回原始影像
# ---------------------------------------------------------

def draw_wireframe_image(image_path: Path, wire_json_path: Path, out_path: Path) -> Path:
    """從 wireframe JSON (inference_results.json 格式) 把 lines_pred 畫在原圖上。"""
    image_path = Path(image_path)
    wire_json_path = Path(wire_json_path)
    out_path = Path(out_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    data = json.loads(wire_json_path.read_text())
    # 單張模式，只會有一個 entry；如果是 list 用第一個
    if isinstance(data, list):
        entry = data[0]
    else:
        entry = data

    lines = np.asarray(entry.get("lines_pred", []), np.float32)
    if lines.size == 0:
        cv2.imwrite(str(out_path), img)
        return out_path

    # 直接畫線
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imwrite(str(out_path), img)
    return out_path


# ---------------------------------------------------------
# 2. Graph 視覺化：畫 bounding box + category
# ---------------------------------------------------------

def draw_graph_image(image_path: Path, graph_entry: dict, out_path: Path) -> Path:
    """從 real graph JSON entry 把 node bbox 畫回原圖。"""
    image_path = Path(image_path)
    out_path = Path(out_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    nodes = graph_entry.get("nodes", [])
    for node in nodes:
        cx, cy, _ = node["center"]
        w = node["w"]
        h = node["h"]
        cat = node["category"]

        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, cat, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(str(out_path), img)
    return out_path


# ---------------------------------------------------------
# 3. Fine localization
# ---------------------------------------------------------

def run_fine_localization(query_img: Path, ref_img: Path, work_dir: Path) -> dict:
    """
    使用 localize_pnp_icp_index.py 對單一 (real, synth_ref) 做 PnP→ICP 定位。

    1) 建一個只包含這組 pair 的 pairs_json（符合 localize_pnp_icp_index 的格式）
    2) 用 subprocess 呼叫 localize_pnp_icp_index.py
    3) 讀 out_dir 裡的 final_poses.csv，取第一列當作這張 query 的 pose
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) 準備 pairs_json（注意這邊 real / ref 都用「絕對路徑」）
    pairs_json_path = work_dir / "pairs_top1.json"
    pairs_obj = {
        "results": {
            str(query_img): [
                {"ref_image": str(ref_img)}
            ]
        }
    }
    with open(pairs_json_path, "w", encoding="utf-8") as f:
        json.dump(pairs_obj, f, indent=2)

    # 2) 呼叫 localize_pnp_icp_index.py
    fine_out_dir = work_dir / "fine_loc"
    fine_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(LOCALIZE_PNP_ICP_SCRIPT),
        "--pairs_json", str(pairs_json_path),
        "--real_rgb_dir", str(query_img.parent),   # real_name 已是絕對路徑，所以實際不會用到這個
        "--real_depth_dir", str(REAL_DEPTH_DIR),
        "--synth_rgb_dir", str(CAND_IMAGE_ROOT),
        "--synth_depth_dir", str(SYNTH_DEPTH_DIR),
        "--isaac_poses_csv", str(ISAAC_POSES_CSV),
        "--out_dir", str(fine_out_dir),
        "--topk_refs", "1",      # 只用 JSON 裡列出的第一個 ref_image
        "--pnp_max_jump_m", "2.0",
        "--icp_max_jump_m", "2.0",
    ]
    # 如果你想要同時輸出 PnP / ICP 可視化，可以加上 "--vis"
    # cmd.append("--vis")

    subprocess.run(cmd, check=True)

    # 3) 讀 final_poses.csv，取第一列
    pose_csv = fine_out_dir / "final_poses.csv"
    if not pose_csv.is_file():
        raise RuntimeError(f"[fine_loc] final_poses.csv not found at {pose_csv}")

    import csv as _csv
    with open(pose_csv, "r", encoding="utf-8") as f:
        rdr = _csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        raise RuntimeError("[fine_loc] final_poses.csv is empty (no estimated pose).")

    row = rows[0]  # 只有一組 pair，所以第一列就是這張 query 的 pose

    pose = {
        "tx_m": float(row["tx_m"]),
        "ty_m": float(row["ty_m"]),
        "tz_m": float(row["tz_m"]),
        "qw": float(row["qw"]),
        "qx": float(row["qx"]),
        "qy": float(row["qy"]),
        "qz": float(row["qz"]),
    }
    return pose


# ---------------------------------------------------------
# 4. 房間 + 相機在 3D 裡的可視化（matplotlib 3D）
# ---------------------------------------------------------

def plot_room_and_camera(room_mesh_path: Path, pose: dict):
    """
    讀 PLY 房間模型 + Isaac world pose，畫在 matplotlib 3D figure 中。
    這裡假設房間 mesh 和 pose 都是同一個 world frame（+X forward, +Z up）。
    """
    mesh = o3d.io.read_triangle_mesh(str(room_mesh_path))
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty mesh: {room_mesh_path}")

    verts = np.asarray(mesh.vertices)
    tx, ty, tz = pose["tx_m"], pose["ty_m"], pose["tz_m"]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # 房間：點雲粗略畫出
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1, alpha=0.3)

    # 相機位置：畫一個大一點的點
    ax.scatter([tx], [ty], [tz], s=80, c="r", marker="o")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Camera pose in room (Isaac world frame)")

    # 調整視角讓 Z 向上
    ax.view_init(elev=20, azim=60)

    # 設定等比例（避免被拉扁）
    max_range = np.array([
        verts[:, 0].max() - verts[:, 0].min(),
        verts[:, 1].max() - verts[:, 1].min(),
        verts[:, 2].max() - verts[:, 2].min()
    ]).max()
    mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
    mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
    mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    return fig


# ---------------------------------------------------------
# 5. 整個 pipeline：上傳 → graph/wireframe → retrieve top-1 → fine loc → 可視化
# ---------------------------------------------------------

def run_full_pipeline(upload_image_path: str):
    """
    給 Gradio 用的主函式。
    輸入：使用者上傳影像的檔案路徑（Gradio Image(type='filepath')）
    輸出：
      - graph_overlay_path (中間 column)
      - wireframe_overlay_path
      - retrieved_top1_path
      - room_camera_fig (右邊 column 的 3D 圖)
    """
    upload_image_path = Path(upload_image_path)
    if not upload_image_path.is_file():
        raise RuntimeError("Uploaded image not found.")

    # --- 建一個新的暫存資料夾 ---
    session_dir = TMP_ROOT / str(uuid.uuid4())
    session_dir.mkdir(parents=True, exist_ok=True)

    # 把上傳的檔案複製到暫存資料夾裡，確保檔名一致
    query_img_path = session_dir / upload_image_path.name
    shutil.copy(upload_image_path, query_img_path)

    # -----------------------------
    # (2) 建立 real graph (single image)
    # -----------------------------
    single_graph_json = session_dir / "single_graph.json"
    graph_entry = build_real_graph_single_image(query_img_path, single_graph_json)

    # build_real_graph_single_image 已經把 JSON 寫到 single_graph.json
    # 但 graph_bear 期待的是「list of entries」，所以再包一層 list 給 query_graph.json
    query_graph_json = session_dir / "query_graph.json"
    with open(query_graph_json, "w", encoding="utf-8") as f:
        json.dump([graph_entry], f, indent=2)

    # Graph 視覺化（中間 column 第一張圖）
    graph_vis_path = session_dir / "graph_overlay.png"
    draw_graph_image(query_img_path, graph_entry, graph_vis_path)

    # -----------------------------
    # 建立 wireframe（用 inference.py, 單張模式）
    # -----------------------------
    # wire_output_dir = session_dir  # 直接把 inference_results.json 丟在 session 資料夾
    # cmd = [
    #     "python",
    #     str(INFERENCE_PY),
    #     str(HAWP_CONFIG),
    #     "--ckpt", str(HAWP_CKPT),
    #     "--image", str(query_img_path),
    #     "--output", str(wire_output_dir),
    # ]
    # subprocess.run(cmd, check=True)

    # wire_query_json = wire_output_dir / "inference_results.json"

    wire_output_dir = session_dir  # inference_results.json 會寫在這裡

    inference.run_hawp_single_image(
        config_path=str(HAWP_CONFIG),
        ckpt_path=str(HAWP_CKPT),
        image_path=str(query_img_path),
        output_dir=str(wire_output_dir),
        j2l=None,
        jhm=None,
        rscale=2,
        seed=42,
    )

    wire_query_json = wire_output_dir / "inference_results.json"

    # Wireframe 視覺化（中間 column 第二張圖）
    wire_vis_path = session_dir / "wireframe_overlay.png"
    draw_wireframe_image(query_img_path, wire_query_json, wire_vis_path)

    # -----------------------------
    # (3) 執行 retrieve（graph_bear_Demo_1127.py, seq_wire + GlueStick re-rank）
    # -----------------------------
    out_root = session_dir / "retrieval"
    out_root.mkdir(exist_ok=True, parents=True)

    # 用「假 argv」呼叫 graph_bear 的 main()
    import sys
    argv_backup = sys.argv[:]
    sys.argv = [
        "graph_bear_Demo_1127.py",
        "--query_json", str(query_graph_json),
        "--candidate_json", str(CAND_GRAPH_JSON),
        "--query_root", str(session_dir),       # query 圖片就放在這個 session 目錄
        "--candidate_root", str(CAND_IMAGE_ROOT),
        "--wire_query_json", str(wire_query_json),
        "--wire_candidate_json", str(WIRE_CAND_JSON),
        "--viz_flag", "seq_wire",
        "--out_root", str(out_root),
        "--fusion", "score",
        "--w_seq", "0.5",
        "--w_wire", "0.5",
    ]
    retr.main()
    sys.argv = argv_backup

    # 讀出 top-1 JSON
    seq_dir = out_root / "seq_wire"
    results_json = seq_dir / "results_seq_wire.json"
    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 只有一個 query，所以直接拿第一個 key
    qname, rows = next(iter(results.items()))
    top1_row = rows[0]
    top1_path = top1_row.get("path") or top1_row.get("name")
    # 用 graph_bear 裡的 _resolve_img_path 補成完整路徑
    from graph_bear_Demo_1127 import _resolve_img_path
    top1_img_abs = Path(_resolve_img_path(top1_path, CAND_IMAGE_ROOT))

    # -----------------------------
    # (4) fine localization → pose
    # -----------------------------
    pose = run_fine_localization(query_img_path, top1_img_abs, session_dir)

    # -----------------------------
    # (5) 房間 + camera pose 視覺化（右邊 column）
    # -----------------------------
    room_fig = plot_room_and_camera(ROOM_MESH_PATH, pose)

    local_top1_path = session_dir / "top1_synth.png"
    shutil.copy(top1_img_abs, local_top1_path)

    # 中間 column 我先回傳 3 張圖：
    #   1) graph overlays
    #   2) wireframe overlays
    #   3) retrieved top-1 image
    return str(graph_vis_path), str(wire_vis_path), str(top1_img_abs), room_fig


# ---------------------------------------------------------
# 6. Gradio 介面：3 個 column
# ---------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# BIM-based Indoor Localization Demo")

    with gr.Row():
        # 左：上傳 + 按鈕
        with gr.Column():
            upload = gr.Image(type="filepath", label="Upload a query image")
            run_btn = gr.Button("Run Localization")

        # 中：graph / wireframe / top-1
        with gr.Column():
            graph_out = gr.Image(label="Real graph overlay")
            wire_out  = gr.Image(label="Wireframe overlay")
            top1_out  = gr.Image(label="Retrieved top-1 synthetic image")

        # 右：3D camera pose in room
        with gr.Column():
            pose_plot = gr.Plot(label="Camera pose in room")

    run_btn.click(
        fn=run_full_pipeline,
        inputs=[upload],
        outputs=[graph_out, wire_out, top1_out, pose_plot],
    )

if __name__ == "__main__":
    demo.launch()
