import os
import json
from pathlib import Path

def extract_unique_paths(input_folder: str, output_txt: str):
    """
    讀取 input_folder 底下所有 .json 檔，
    收集所有非空且不等於 'INVALID' 的 value（完整路徑字串），
    去重、排序後寫入 output_txt。
    """
    unique_paths = set()

    for json_path in Path(input_folder).glob("*.json"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for val in data.values():
            if not val or val.upper() == "INVALID":
                continue
            unique_paths.add(val)

    # 寫入 txt
    with open(output_txt, 'w', encoding='utf-8') as f:
        for path in sorted(unique_paths):
            f.write(path + "\n")

    print(f"共找到 {len(unique_paths)} 種不同的路徑，已存至 {output_txt}")

if __name__ == "__main__":
    input_folder = "/D/hoa/Delta_project/Dataset_0615/instance_segmentation/"
    output_txt   = "0615_classes.txt"
    extract_unique_paths(input_folder, output_txt)
