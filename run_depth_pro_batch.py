#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批次使用 Apple Depth Pro 預測某資料夾底下所有影像的深度，
並將結果以 .npy 存到指定輸出資料夾。

命名規則：
    <原始影像檔名> + ".npy"
例如：
    DSC09508.JPG → DSC09508.JPG.npy

這樣就可以讓 construct_real_graph.py 依照
    depth_filename = Path(fn).name + '.npy'
順利讀到對應的深度圖。
"""

import argparse
from pathlib import Path
import torch

import numpy as np
from tqdm import tqdm

import depth_pro  # 來自 https://github.com/apple/ml-depth-pro


def is_image_file(path: Path) -> bool:
    """判斷是否為常見圖片格式。"""
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main(input_dir: str, output_dir: str, overwrite: bool = False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 建立 Depth Pro model 與前處理 transform
    # 依官方 README：
    #   model, transform = depth_pro.create_model_and_transforms()
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # 收集所有圖片檔（含子資料夾）
    image_paths = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and is_image_file(p)]
    )

    if not image_paths:
        print(f"[Warning] No images found under: {input_dir}")
        return

    print(f"[Info] Found {len(image_paths)} images under {input_dir}")
    print(f"[Info] Saving depth maps to {output_dir}")

    for img_path in tqdm(image_paths, desc="Running Depth Pro"):
        # construct_real_graph.py 使用 Path(fn).name + ".npy"
        # 所以這邊一定要用「檔名本身」而不是相對路徑或絕對路徑。
        depth_filename = img_path.name + ".npy"
        depth_path = output_dir / depth_filename

        if depth_path.exists() and not overwrite:
            # 已存在就略過，避免重算
            continue

        # 使用官方 helper 讀圖，會回傳 image（PIL）、原始大小資訊與 f_px
        image, _, f_px = depth_pro.load_rgb(str(img_path))
        image = transform(image)

        # 推論深度
        with torch.no_grad():  # 如果沒有這個 context，可以改用 torch.no_grad()
            prediction = model.infer(image, f_px=f_px)

        depth = prediction["depth"]
        # depth 可能是 torch.Tensor，轉成 numpy array 存檔
        try:
            depth_np = depth.cpu().numpy()
        except AttributeError:
            # 如果本來就是 numpy，就直接用
            depth_np = np.asarray(depth)

        # 儲存成 .npy 檔
        np.save(str(depth_path), depth_np)

    print("[Done] All depth maps saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch run Apple Depth Pro on all images in a folder "
                    "and save depth maps as .npy files."
    )
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="輸入圖片所在資料夾（會遞迴搜尋子資料夾）"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="輸出深度圖要存放的資料夾（需與 construct_real_graph.py 的 DEPTH_FOLDER 一致）"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若指定，已存在的 .npy 檔會被覆寫"
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, overwrite=args.overwrite)
