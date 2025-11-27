import os
import json
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from pycocotools import mask as mask_util

from ultralytics import YOLO
import torch
import depth_pro  # Apple Depth Pro 官方套件
from depth_pro.depth_pro import DepthProConfig
from typing import Union, Optional



import utils.utils as utils  # reuse build_scene_graph_from_components, graph_descriptor, graph_to_dict

REAL_FOLDER      = Path("/D/lulu/Delta/Image_data/real_image_readingRoom/")
# COCO_JSON        = Path("/D/lulu/home/Delta/graph/YOLO_anno/readingRoom/_annotations_train.coco.json")
OUTPUT_JSON_REAL = Path("/D/lulu/Delta/graph/readingRoom/readingRoom_graph_real_train_single.json")
OUTPUT_VIS_REAL  = Path("/D/lulu/Delta/graph/readingRoom/vis_single/")
DEPTH_FOLDER     = Path("/D/lulu/Delta/Image_data/depth_maps/readingRoom_depth_single/")
CAMERA_INTRINSICS = np.array([1649.5450819708324, 1649.719272635186, 1200.0,  673.5], dtype=float)


STRUCTURAL_CLASSES = [
    "Walls", "Columns", "Beams", "Ceilings", "Floors", "Doors", "Windows", "Pipe", "Cable_Tray"
]
REAL2SYN = {
    "wall":        "Walls",
    "column":      "Columns",
    "beam":        "Beams",
    "floor":        "Floors",
    "door":        "Doors",
    "window":      "Windows",
    "pipe":        "Pipe",
    "cable":       "Cable_Tray",
    "ceiling":     "Ceilings",    # 對應 ceiling_0, ceiling_1, ceilling_0
    "ceilling":    "Ceilings"  # 處理拼字錯誤
}

_base_rules = {
    ("Doors","Walls"):"door_in_wall",
    ("Windows","Walls"):"window_in_wall",
    ("Beams","Walls"):"beam_on_wall",
    ("Beams","Windows"):"beam_above_window",
    ("Ceilings","Walls"):"ceiling_on_wall",
    ("Columns","Ceilings"):"column_supports_ceiling",
    ("Beams","Ceilings"):"beam_supports_ceiling",
    ("Columns","Floors"):"column_on_slab",
    ("Walls","Floors"):"wall_on_slab",
    ("Walls","Columns"):"wall_column_conn",
    ("Walls","Walls"):"wall_adjacent",
}

# === 單張模式：YOLO + Depth Pro 模型初始化 ===
# YOLO segmentation 權重檔，請改成你實際 best.pt 的路徑
YOLO_WEIGHTS = Path("/D/lulu/home/Delta/graph/YOLO/best.pt")  # 例如 Path("/D/lulu/Delta/graph/best.pt")

# 只初始化一次，之後重複拿來用
_yolo_model = YOLO(str(YOLO_WEIGHTS))


def _create_depth_model_and_transform():
    """
    建立 Apple Depth Pro model 與前處理 transform，
    並用自訂的 checkpoint_uri（絕對路徑）載入權重。
    """
    # 請把這個路徑改成你實際的 depth_pro.pt 位置
    # 例如：/home/lulu/ml-depth-pro/checkpoints/depth_pro.pt
    ckpt_path = "/D/lulu/home/ml-depth-pro/checkpoints/depth_pro.pt"

    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=ckpt_path,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )

    # 這裡可以選擇要不要用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 用自訂的 config 建 model
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=device,
        precision=torch.float32,  # 需要的話你也可以改成 torch.half
    )
    model.eval()
    return model, transform

# def _create_depth_model_and_transform():
#     """建立 Apple Depth Pro model 與前處理 transform。"""
#     model, transform = depth_pro.create_model_and_transforms()
#     model.eval()
#     return model, transform

# 同樣在載入時建好一次
_depth_model, _depth_transform = _create_depth_model_and_transform()


RELATION_RULES = {tuple(sorted(k)):v for k,v in _base_rules.items()}

ANGLE_BINS, DIST_BINS = 4, 4
EDGE_DIM = ANGLE_BINS * DIST_BINS
BBOX_PAD = 0.1

# coco = json.load(open(COCO_JSON, 'r', encoding='utf-8'))
# # image_id -> filename (from extra["name"])
# images_map = {
#     img['id']: img.get('extra',{}).get('name')
#     for img in coco['images']
#     if img.get('extra',{}).get('name') and 
#        (REAL_FOLDER / img['extra']['name']).exists()
# }

# # # --- 修改開始 ---
# # # 1. 先建立一個 filename -> image_id 的映射，這樣可以確保每個檔名只對應一個 ID (去除重複檔案)
# # filename_to_id = {}
# # for img in coco['images']:
# #     fn = img.get('extra', {}).get('name')
# #     # 檢查檔名存在且實體檔案存在
# #     if fn and (REAL_FOLDER / fn).exists():
# #         # 如果檔名已存在，這行會覆蓋舊的 ID，確保一對一 (或者你可以選擇 if fn not in filename_to_id 才賦值)
# #         filename_to_id[fn] = img['id']

# # # 2. 反轉回來變成 image_id -> filename，供後續程式使用
# # images_map = {
# #     img_id: fn
# #     for fn, img_id in filename_to_id.items()
# # }
# # # --- 修改結束 ---

# orig_size_map = {
#     img['id']: (img['width'], img['height'])
#     for img in coco['images']
#     if img['id'] in images_map
# }

# # category_id -> raw category name
# cat_map = {c['id']: c['name'] for c in coco['categories']}
# # group annotations by image
# from collections import defaultdict
# anns_by_img = defaultdict(list)
# for ann in coco['annotations']:
#     if ann['image_id'] in images_map:
#         anns_by_img[ann['image_id']].append(ann)




def build_real_graph_single_image(image_path: Union[str, Path],
                                  output_json_path: Union[str, Path]) -> Optional[dict]:
    """
    單張圖片流程：
      1) 用 Depth Pro 推深度（在記憶體中，不寫 .npy）
      2) 用 YOLO segmentation 取得 instance masks / boxes / classes
      3) 組 comps → build_scene_graph_from_components → graph_to_dict
      4) 將結果存成一個 JSON 檔
    """
    image_path = Path(image_path)
    output_json_path = Path(output_json_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- 讀 RGB 原圖 ---
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_orig.shape[:2]

    # --- Depth Pro: 單張推論，結果放在記憶體 ---
    # 這邊直接沿用 run_depth_pro_batch.py 的寫法
    image_pil, _, f_px = depth_pro.load_rgb(str(image_path))
    image_tensor = _depth_transform(image_pil)

    with torch.no_grad():
        prediction = _depth_model.infer(image_tensor, f_px=f_px)

    depth = prediction["depth"]
    try:
        depth_np = depth.cpu().numpy()
    except AttributeError:
        depth_np = np.asarray(depth)

    # --- 統一 resize 到 (W, H) ---
    target_size = (1920, 1080)  # (W, H)
    W, H = target_size
    scale_x = W / orig_w
    scale_y = H / orig_h

    rgb = cv2.resize(rgb_orig, target_size, interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth_np, target_size, interpolation=cv2.INTER_NEAREST)

    dmin, dmax = float(depth_resized.min()), float(depth_resized.max())
    depth_norm = (depth_resized - dmin) / (dmax - dmin + 1e-6)

    # --- YOLO segmentation ---
    # 注意：這邊直接餵路徑，ultralytics 會自動讀圖
    results = _yolo_model(str(image_path))
    res = results[0]

    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        print(f"[Warning] YOLO 沒有在 {image_path.name} 偵測到任何 instance，跳過。")
        return None

    masks_data = res.masks.data.cpu().numpy()      # (N, Hm, Wm)
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()      # (N, 4)
    cls_ids    = res.boxes.cls.cpu().numpy().astype(int)

    # 取得類別名稱對照
    names = getattr(res, "names", None)
    if names is None:
        names = _yolo_model.names  # dict: id -> name

    comps = []

    for mask, box, cid in zip(masks_data, boxes_xyxy, cls_ids):
        raw_name = names[int(cid)]
        base = raw_name.split('_')[0].lower()
        struct = REAL2SYN.get(base)

        # 和原本 REAL2SYN + STRUCTURAL_CLASSES 的邏輯一樣
        if struct not in STRUCTURAL_CLASSES:
            continue

        x1, y1, x2, y2 = box
        # 先假設 YOLO 的座標是在原始影像解析度
        x = x1 * scale_x
        y = y1 * scale_y
        w = (x2 - x1) * scale_x
        h = (y2 - y1) * scale_y
        cx = x + w / 2.0
        cy = y + h / 2.0

        # 將 YOLO 的 mask resize 到 (W, H)，再二值化
        mask_resized = cv2.resize(mask.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized > 0.5
        if not mask_bool.any():
            continue

        comps.append({
            "category": struct,
            "bbox": (float(x), float(y), float(w), float(h)),
            "center": (float(cx) / W, float(cy) / H),
            "mask": mask_bool,
        })

    if not comps:
        print(f"[Warning] {image_path.name} 雖然有偵測到，但沒有任何符合 STRUCTURAL_CLASSES 的 component。")
        return None

    # --- 建 scene graph ---
    g = utils.build_scene_graph_from_components(
        comps, W, H, depth_norm, RELATION_RULES, BBOX_PAD, CAMERA_INTRINSICS
    )
    if g.number_of_nodes() == 0:
        print(f"[Warning] scene graph for {image_path.name} 沒有節點，跳過。")
        return None

    # --- 計算 descriptor ---
    vec = utils.graph_descriptor(
        g, STRUCTURAL_CLASSES, EDGE_DIM, ANGLE_BINS, DIST_BINS
    )

    # 如果你想產生可視化，也可以在這裡呼叫 save_real_vis，這邊先註解掉：
    # utils.save_real_vis(image_path.name, rgb, comps, g, Path("vis_single"))

    # --- 組成輸出 dict 並寫成 JSON 檔 ---
    graph_entry = utils.graph_to_dict(g, image_path.name, vec)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(graph_entry, f, indent=2)

    print(f"[✓] Saved real graph for {image_path.name} to {output_json_path}")
    return graph_entry


def process_real(image_id: int) -> dict:
    fn = images_map[image_id]
    img_path = REAL_FOLDER / fn
    orig_w, orig_h = orig_size_map[image_id]

    rgb_orig = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)


    depth_filename = Path(fn).name + '.npy'
    depth_path     = DEPTH_FOLDER / depth_filename
    try:
        depth_map = np.load(str(depth_path))
    except FileNotFoundError:
        print(f"[Warning] depth not found for {fn}, skipping.")
        return None


    target_size = (1920, 1080)  # (W, H)
    W, H = target_size
    scale_x = W / orig_w
    scale_y = H / orig_h



    rgb  = cv2.resize(rgb_orig,  target_size, interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_NEAREST)

    dmin, dmax = float(depth.min()), float(depth.max())
    depth = (depth - dmin) / (dmax - dmin + 1e-6)



    comps = []
    for ann in anns_by_img[image_id]:
        raw = cat_map[ann['category_id']]
        base = raw.split('_')[0].lower()
        struct = REAL2SYN.get(base)
        if struct not in STRUCTURAL_CLASSES:
            continue

        x,y,w,h = ann['bbox']
        x *= scale_x;  y *= scale_y
        w *= scale_x;  h *= scale_y
        cx = x + w / 2
        cy = y + h / 2
        # create bbox mask for area ratio (optional)
        mask_uint8 = np.zeros((H, W), dtype=np.uint8)
        segs = ann.get('segmentation', [])
        # for seg in segs:
        #     pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        #     pts[:, 0] *= scale_x
        #     pts[:, 1] *= scale_y
        #     cv2.fillPoly(mask_uint8, [pts.astype(np.int32)], 1)
        # mask = mask_uint8.astype(bool)  # 之後方便用 boolean indexing

        mask_uint8 = np.zeros((H, W), dtype=np.uint8)
        segs = ann.get('segmentation', [])
        for seg in segs:
            if isinstance(seg, dict):
                # RLE 格式
                m = mask_util.decode(seg)     # 產生 (H, W) 二值 mask :contentReference[oaicite:5]{index=5}
                mask_uint8 |= m.astype(np.uint8)
            else:
                # 多邊形頂點
                pts = np.array(seg, dtype=np.float32).reshape(-1,2)
                pts[:,0] *= scale_x; pts[:,1] *= scale_y
                cv2.fillPoly(mask_uint8, [pts.astype(np.int32)], 1)
        mask = mask_uint8.astype(bool)        
        if not mask.any():
            continue
        comps.append({
            'category': struct,
            'bbox': (x,y,w,h),
            'center': (cx/W, cy/H),
            'mask': mask
        })

    if not comps:
        return None

    # build scene graph
    g = utils.build_scene_graph_from_components(
        comps, W, H, depth, RELATION_RULES, BBOX_PAD, CAMERA_INTRINSICS
    )
    if g.number_of_nodes()==0:
        return None

    # descriptor
    vec = utils.graph_descriptor(
        g, STRUCTURAL_CLASSES, EDGE_DIM, ANGLE_BINS, DIST_BINS
    )

    # save visualization
    utils.save_real_vis(fn, rgb, comps, g, OUTPUT_VIS_REAL)

    # return JSON entry
    return utils.graph_to_dict(g, fn, vec)


def main_real():
    results = []
    image_ids = list(images_map.keys())
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = { exe.submit(process_real, img_id): img_id for img_id in image_ids }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Processing Real"):
            res = future.result()
            if res: results.append(res)

    with open(OUTPUT_JSON_REAL, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved {len(results)} real graphs to {OUTPUT_JSON_REAL}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a real scene graph for a single image using YOLO seg + Depth Pro."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="要建立 real graph 的影像路徑",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="輸出 graph JSON 路徑（預設為 <image_stem>.graph.json）",
    )

    args = parser.parse_args()
    img_path = Path(args.image)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = img_path.with_suffix(".graph.json")

    build_real_graph_single_image(img_path, out_path)