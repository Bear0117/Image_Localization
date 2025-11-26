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

import utils.utils as utils  # reuse build_scene_graph_from_components, graph_descriptor, graph_to_dict

REAL_FOLDER      = Path("/D/hoa/Delta_project/Dataset_0605_Reading/rgb/")
COCO_JSON        = Path("/D/lulu/home/Delta/graph/YOLO_anno/readingRoom/readingRoom_annotations.coco.json")
OUTPUT_JSON_REAL = Path("/D/lulu/Delta/graph/readingRoom/r611_sim.json")
OUTPUT_VIS_REAL  = Path("/D/lulu/Delta/graph/readingRoom/vis_sim/")
DEPTH_FOLDER     = Path("/D/lulu/Delta/Image_data/depth_maps/readingRoom_depth/")
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

RELATION_RULES = {tuple(sorted(k)):v for k,v in _base_rules.items()}

ANGLE_BINS, DIST_BINS = 4, 4
EDGE_DIM = ANGLE_BINS * DIST_BINS
BBOX_PAD = 0.1

coco = json.load(open(COCO_JSON, 'r', encoding='utf-8'))
# image_id -> filename (from extra["name"])
images_map = {
    img['id']: img.get('extra',{}).get('name')
    for img in coco['images']
    if img.get('extra',{}).get('name') and 
       (REAL_FOLDER / img['extra']['name']).exists()
}
orig_size_map = {
    img['id']: (img['width'], img['height'])
    for img in coco['images']
    if img['id'] in images_map
}

# category_id -> raw category name
cat_map = {c['id']: c['name'] for c in coco['categories']}
# group annotations by image
from collections import defaultdict
anns_by_img = defaultdict(list)
for ann in coco['annotations']:
    if ann['image_id'] in images_map:
        anns_by_img[ann['image_id']].append(ann)


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
    main_real()