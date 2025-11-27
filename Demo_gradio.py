import os
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

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
CAND_GRAPH_JSON = Path("/D/.../your_candidate_graphs.json")      # synth graph JSON
CAND_IMAGE_ROOT = Path("/D/.../Dataset_0605_Reading/rgb/")       # synth RGB root
WIRE_CAND_JSON  = Path("/D/.../wireframe_candidate.json")        # wireframe DB for candidates

# HAWP / wireframe inference
HAWP_CONFIG = Path("/D/.../configs/wireframe.yaml")
HAWP_CKPT   = Path("/D/.../hawp_checkpoint.pth")
INFERENCE_PY = Path("/D/.../inference.py")  # 你剛剛改過的 inference.py

# 房間 mesh（建議 PLY）
ROOM_MESH_PATH = Path("/D/.../room_mesh.ply")

# 如果 fine localization 是跑某個 script（例如 localize_pnp_icp_index.py）
# 可以在這裡先記錄 script 路徑
FINE_LOC_SCRIPT = Path("/D/.../localize_pnp_icp_index.py")


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
# 3. Fine localization：這裡只留一個入口給你接現有 pipeline
# ---------------------------------------------------------

def run_fine_localization(query_img: Path, ref_img: Path, work_dir: Path) -> dict:
    """
    TODO: 這裡請接你原本的 fine localization pipeline。
    建議做法：
      1) 在 work_dir 下產生 pairs_json (只包含這一組 query/ref pair)。
      2) 用 subprocess 呼叫 localize_pnp_icp_index.py。
      3) 讀它輸出的 pose CSV 檔（格式跟 final_poses_hallway.csv 一樣）。
      4) 取最佳那一列，回傳成 dict。

    下面是一個「假設已經有 pose.csv」的範例，你只要把 path 改成你 script 的輸出即可。
    """
    # === 範例：假設 script 把結果寫到 work_dir / "pose.csv" ===
    pose_csv = work_dir / "pose.csv"

    # 你真正的 code 應該是像這樣：
    # cmd = [
    #     "python", str(FINE_LOC_SCRIPT),
    #     "--pairs_json", str(pairs_json),
    #     "--real_rgb_dir", ...,
    #     "--synth_rgb_dir", ...,
    #     "--output_csv", str(pose_csv),
    #     ...
    # ]
    # subprocess.run(cmd, check=True)

    if not pose_csv.is_file():
        # 這裡先給一個假 pose，避免 demo 爆炸，你接好 script 後可以刪掉整段 if。
        print("[WARN] pose.csv not found, using dummy pose at origin.")
        return {
            "tx_m": 0.0,
            "ty_m": 0.0,
            "tz_m": 0.0,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        }

    import pandas as pd
    df = pd.read_csv(pose_csv)
    # 假設第一列就是最佳 pose
    row = df.iloc[0]

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
    wire_output_dir = session_dir  # 直接把 inference_results.json 丟在 session 資料夾
    cmd = [
        "python",
        str(INFERENCE_PY),
        str(HAWP_CONFIG),
        "--ckpt", str(HAWP_CKPT),
        "--image", str(query_img_path),
        "--output", str(wire_output_dir),
    ]
    subprocess.run(cmd, check=True)

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
