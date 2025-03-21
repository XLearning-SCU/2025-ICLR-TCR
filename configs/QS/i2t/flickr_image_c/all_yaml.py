import os
import yaml

# 目标文件夹
folder_path = "/xlearning/haobin/project/TCR/release/configs/blip_flickr_coco_c/i2t_large/flickr_image_c"  # 替换为你的 YAML 文件所在目录

# 替换规则
replacements = {
    "/flickr-TP/flickr_annotations": "/flickr/flickr_annotations",
    "/data/haobin/tta_retrieval/data/retireval_tta": "./dataset"
}

# 遍历文件夹中的所有 YAML 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".yaml"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 YAML 文件
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 进行字符串替换
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # 将修改后的内容写回文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Updated {filename}")
