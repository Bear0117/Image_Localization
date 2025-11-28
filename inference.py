import torch
from hawp.base.utils.comm import to_device
from hawp.base.utils.logger import setup_logger
#from hawp.base.utils.checkpoint import DetectronCheckpointer  # Not used in inference
#from hawp.base.utils.metric_evaluation import TPFP, AP  # Not needed for simple inference

from hawp.fsl.config import cfg
#from hawp.fsl.config.paths_catalog import DatasetCatalog  # Not used for inference
#from hawp.fsl.dataset import build_test_dataset  # Not used
from hawp.fsl.model.build import build_model
from hawp.fsl.dataset.build import build_transform

import os
import os.path as osp
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np
import importlib
import time
import random
from PIL import Image

def run_hawp_single_image(
    config_path,
    ckpt_path,
    image_path,
    output_dir,
    j2l=None,
    jhm=None,
    rscale=2,
    seed=42,
):
    """
    單張圖片版本的 HAWP inference：
      - config_path: plnet.yaml 等 config 檔路徑
      - ckpt_path:   plnet.pth 等 checkpoint 路徑
      - image_path:  要做 wireframe 的圖片 (str or Path)
      - output_dir:  輸出資料夾 (會寫 inference_results.json 到這裡)
    """
    import os
    import os.path as osp

    # 1) merge config & 準備 output 資料夾
    cfg.merge_from_file(str(config_path))
    root = output_dir if output_dir is not None else osp.dirname(str(ckpt_path))
    os.makedirs(root, exist_ok=True)

    # 2) random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = cfg.MODEL.DEVICE

    # 3) 建 model
    model = build_model(cfg)
    model = model.to(device)
    if rscale is not None:
        model.use_residual = rscale
    if j2l is not None:
        model.j2l_threshold = j2l
    if jhm is not None:
        model.jhm_threshold = jhm

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.eval()

    # 4) transform
    transform = build_transform(cfg)

    # 5) 單張圖片 inference
    image_path = str(image_path)
    logger = setup_logger("hawp.inference.single", root)

    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to open image {image_path}: {e}")
        return None

    w, h = im.size
    meta = {"filename": image_path, "width": w, "height": h}
    image_np = np.array(im)
    tensor = transform(image_np)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        try:
            output, extra_info = model(tensor.to(device), [meta])
        except IndexError as e:
            logger.info(f"Skipping image {image_path} due to error: {e}")
            return None

    # 檢查 / 轉成 list
    if isinstance(output["lines_pred"], torch.Tensor):
        if output["lines_pred"].numel() == 0:
            logger.info(f"Skipping image {image_path} due to zero predictions")
            return None
        for k in list(output.keys()):
            if isinstance(output[k], torch.Tensor):
                output[k] = output[k].tolist()
    else:
        if len(output["lines_pred"]) == 0:
            logger.info(f"Skipping image {image_path} due to zero predictions")
            return None

    results = [output]
    outpath = osp.join(root, "inference_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f)

    logger.info(f"Saved inference results to {outpath}")
    return outpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HAWP Inference')
    
    parser.add_argument('config', help='Path to the config file')
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--imagedir", type=str, default=None,
                        help="Directory containing images for inference")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path for inference")
    parser.add_argument("--j2l", default=None, type=float, help='Threshold for junction-line attraction')
    parser.add_argument("--jhm", default=None, type=float, help='Threshold for junction heatmap')
    parser.add_argument("--rscale", default=2, type=int, help='Residual scale')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output', default=None, help='Path for saving outputs')
    
    args = parser.parse_args()

    # Merge config file
    config_path = args.config
    cfg.merge_from_file(config_path)
    root = args.output
    if root is None:
        root = os.path.dirname(args.ckpt)
    os.makedirs(root, exist_ok=True)

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = cfg.MODEL.DEVICE

    # Build and prepare the model
    model = build_model(cfg)
    model = model.to(device)
    if args.rscale is not None:
        model.use_residual = args.rscale
    if args.j2l:
        model.j2l_threshold = args.j2l
    if args.jhm:
        model.jhm_threshold = args.jhm

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.eval()

    # Build image transformation
    transform = build_transform(cfg)

    # 決定要用「單張圖片」還是「資料夾」模式
    image_extensions = ['.jpg', '.jpeg', '.png']

    if args.image is not None:
        # 單張圖片模式
        ext = osp.splitext(args.image)[1].lower()
        if ext not in image_extensions:
            print("Provided --image is not a supported format:", args.image)
            exit(1)
        image_files = [args.image]
    elif args.imagedir is not None:
        # 資料夾模式（跟原本一樣）
        image_files = [
            osp.join(args.imagedir, f)
            for f in os.listdir(args.imagedir)
            if osp.splitext(f)[1].lower() in image_extensions
        ]
        if not image_files:
            print("No images found in the directory:", args.imagedir)
            exit(0)
    else:
        print("You must specify either --image or --imagedir")
        exit(1)

    logger = setup_logger('hawp.inference', root)
    logger.info("Running inference on {} images".format(len(image_files)))
    
    results = []
    for image_path in tqdm(image_files):
        try:
            im = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open image {}: {}".format(image_path, e))
            continue
        w, h = im.size
        meta = {"filename": image_path, "width": w, "height": h}
        image_np = np.array(im)
        tensor = transform(image_np)
        tensor = tensor.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            try:
                output, extra_info = model(tensor.to(device), [meta])
            except IndexError as e:
                logger.info("Skipping image {} due to error: {}".format(image_path, e))
                continue

        # Additional check: Skip image if 'lines_pred' is empty
        if isinstance(output['lines_pred'], torch.Tensor):
            if output['lines_pred'].numel() == 0:
                logger.info("Skipping image {} due to zero predictions".format(image_path))
                continue
            for k in output.keys():
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
        else:
            if len(output['lines_pred']) == 0:
                logger.info("Skipping image {} due to zero predictions".format(image_path))
                continue

        results.append(output)

    
    outpath = osp.join(root, "inference_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f)
    logger.info("Saved inference results to {}".format(outpath))
