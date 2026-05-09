# ==== 手动改这里 ====
GT_DIR   = "/data/baiyixue/CAD/screenshots/00002_index_1/step0/3D_isometric.png"         # GT图目录或单张图片
PRED_DIR = ("/data/baiyixue/CAD/screenshots/00002_index_1/step0/3D_top.png")       # 预测图目录或单张图片
OUT_CSV  = "/home/baiyixue/project/op-cad/inference/inference_results/test/out_dinov2_cosine.csv"

# 兼容你原先的模型名：dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14
MODEL_NAME = "dinov2_vitb14" 
IMAGE_SIZE = 518
USE_FIVE_CROP = True
DEVICE = "cuda"   # 或 "cpu"

# ==== 依赖 ====
import os, glob
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd

# ==== HF DINOv2 ====
# pip install transformers pillow torch torchvision
from transformers import Dinov2Model

HF_NAME_MAP = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant"
}
HF_MODEL_ID = HF_NAME_MAP.get(MODEL_NAME, "facebook/dinov2-large")

print(f"[INFO] Using HF model: {HF_MODEL_ID} (mapped from {MODEL_NAME})")

model = Dinov2Model.from_pretrained(HF_MODEL_ID)
model.eval().to(DEVICE)

# ==== 预处理（与ImageNet一致，等价于processor默认）====
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))

to_tensor_norm = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize
])

def five_crop_tensor(img):
    base = transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)
    t5 = transforms.FiveCrop(IMAGE_SIZE)
    img = base(img)
    crops = t5(img)  # 5 PIL
    crops = [normalize(transforms.ToTensor()(c)) for c in crops]
    return torch.stack(crops, dim=0)  # (5,3,H,W)

def load_image_tensor(path, use_five_crop=True):
    img = Image.open(path).convert("RGB")
    if use_five_crop:
        return five_crop_tensor(img)  # (5,3,H,W)
    else:
        return to_tensor_norm(img).unsqueeze(0)  # (1,3,H,W)

# ==== 特征提取 ====
@torch.no_grad()
def extract_feat(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    img_tensor: (N,3,H,W) CPU tensor
    Returns: (D,) L2归一化后的向量（对N个crop做均值池化）
    """
    img_tensor = img_tensor.to(DEVICE)
    outputs = model(pixel_values=img_tensor)
    # 取 CLS（第0个token）作为全局特征
    feats = outputs.last_hidden_state[:, 0, :]  # (N, D)
    feats = feats.mean(dim=0, keepdim=True)     # (1, D) 平均池化5-crop
    feats = F.normalize(feats, dim=1)
    return feats.squeeze(0).detach().cpu()

# ==== 既可传“目录”也可传“单图” ====
def index_images(path_or_dir):
    def is_img(p):
        ext = os.path.splitext(p)[1].lower()
        return ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    if os.path.isfile(path_or_dir):
        if not is_img(path_or_dir):
            return {}
        k = os.path.splitext(os.path.basename(path_or_dir))[0]
        return {k: path_or_dir}
    else:
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(path_or_dir, e))
        return {os.path.splitext(os.path.basename(p))[0]: p for p in files}

def build_pairs(gt_input, pred_input):
    gt_is_file, pred_is_file = os.path.isfile(gt_input), os.path.isfile(pred_input)
    gt_is_dir,  pred_is_dir  = os.path.isdir(gt_input), os.path.isdir(pred_input)

    gt_map = index_images(gt_input)
    pred_map = index_images(pred_input)

    # Case A: 两边都是单文件 -> 直接配对（不要求同名）
    if gt_is_file and pred_is_file:
        gt_path  = list(gt_map.values())[0]
        pred_path = list(pred_map.values())[0]
        return {"pair0": (gt_path, pred_path)}

    # Case B: 单文件 vs 目录 -> 用单文件的基名在目录中找
    if gt_is_file and pred_is_dir:
        k = next(iter(gt_map.keys()))
        if k in pred_map:
            return {k: (gt_map[k], pred_map[k])}
        raise RuntimeError(f"[ERROR] 目录中未找到与 {k} 同名的预测图：{pred_input}")

    if gt_is_dir and pred_is_file:
        k = next(iter(pred_map.keys()))
        if k in gt_map:
            return {k: (gt_map[k], pred_map[k])}
        raise RuntimeError(f"[ERROR] 目录中未找到与 {k} 同名的GT图：{gt_input}")

    # Case C: 目录 vs 目录 -> 取同名交集
    keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    return {k: (gt_map[k], pred_map[k]) for k in keys}

pairs = build_pairs(GT_DIR, PRED_DIR)
if not pairs:
    raise RuntimeError(f"[ERROR] 未找到可配对图片。GT={GT_DIR} PRED={PRED_DIR}")

print(f"[INFO] Pairs: {len(pairs)}")


# ==== 主循环 ====
rows = []
for k, (gt_path, pred_path) in pairs.items():
    gt_tensor   = load_image_tensor(gt_path, use_five_crop=USE_FIVE_CROP)
    pred_tensor = load_image_tensor(pred_path, use_five_crop=USE_FIVE_CROP)

    gt_feat   = extract_feat(gt_tensor)
    pred_feat = extract_feat(pred_tensor)

    cos_sim  = F.cosine_similarity(gt_feat.unsqueeze(0), pred_feat.unsqueeze(0)).item()
    cos_dist = 1.0 - cos_sim

    rows.append({
        "key": k,
        "gt_path": gt_path,
        "pred_path": pred_path,
        "cosine_similarity": cos_sim,
        "cosine_distance": cos_dist,
        "model": HF_MODEL_ID,
        "image_size": IMAGE_SIZE,
        "five_crop": int(USE_FIVE_CROP),
        "device": DEVICE
    })

df = pd.DataFrame(rows).sort_values("cosine_distance")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Done. Saved to: {OUT_CSV}\nTop-5 most similar:")
print(df.head(5).to_string(index=False))
