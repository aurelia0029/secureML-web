#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ======================================================
# 自動偵測路徑設定 (取代原本寫死的 /Users/ 或 /home/)
# ======================================================
# 取得目前此腳本所在的資料夾路徑
PROJECT_ROOT = Path(__file__).resolve().parent

# 將專案根目錄加入 sys.path，這樣才能 import models 和 tasks
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 統一在上方進行 import
try:
    from tasks.ppmodel import PPNet
    from models.model_protopnet.receptive_field import compute_proto_layer_rf_info_v2
    from models.model_protopnet.resnet_features import (
        resnet18_features, resnet34_features, resnet50_features,
        resnet101_features, resnet152_features,
    )
    from models.model_protopnet.densenet_features import (
        densenet121_features, densenet161_features, densenet169_features, densenet201_features,
    )
    from models.model_protopnet.vgg_features import (
        vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features,
        vgg16_features, vgg16_bn_features, vgg19_features, vgg19_bn_features,
    )
except ImportError as e:
    print(f"[ERROR] 無法載入模組，請確保此腳本放在專案根目錄下。詳細錯誤: {e}")
    sys.exit(1)


def build_ppnet_model(ckpt_params):
    """
    根據 Checkpoint 參數建立 PPNet 模型
    """
    base_architecture_to_features = {
        "resnet18": resnet18_features,
        "resnet34": resnet34_features,
        "resnet50": resnet50_features,
        "resnet101": resnet101_features,
        "resnet152": resnet152_features,
        "densenet121": densenet121_features,
        "densenet161": densenet161_features,
        "densenet169": densenet169_features,
        "densenet201": densenet201_features,
        "vgg11": vgg11_features,
        "vgg11_bn": vgg11_bn_features,
        "vgg13": vgg13_features,
        "vgg13_bn": vgg13_bn_features,
        "vgg16": vgg16_features,
        "vgg16_bn": vgg16_bn_features,
        "vgg19": vgg19_features,
        "vgg19_bn": vgg19_bn_features,
    }

    base_arch = ckpt_params.get("base_architecture", "vgg19")
    if base_arch not in base_architecture_to_features:
        raise ValueError(f"Unsupported base_architecture for PPNet: {base_arch}")

    img_size = int(ckpt_params.get("img_size", 32))
    num_classes = int(ckpt_params.get("num_classes", 2))
    
    # 處理 prototype_shape，確保是 tuple
    prototype_shape = ckpt_params.get("prototype_shape", (20, 8, 1, 1))
    if isinstance(prototype_shape, list):
        prototype_shape = tuple(prototype_shape)

    # 建立特徵提取層並計算感受野 (RF) 資訊
    features = base_architecture_to_features[base_arch](pretrained=False)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2],
    )

    return PPNet(
        features=features,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=ckpt_params.get("prototype_activation_function", "log"),
        add_on_layers_type=ckpt_params.get("add_on_layers_type", "regular"),
    )


def main():
    parser = argparse.ArgumentParser(description="Inference for PPNet (mprobe_m_219).")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=PROJECT_ROOT / "mprobe_40_model.tar",
        help="Path to model checkpoint (.pt.tar).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "inference_img" / "IMG_9799.JPG",
        help="Path to input image.",
    )
    args = parser.parse_args()

    # 檢查路徑是否存在
    if not args.ckpt.exists():
        raise SystemExit(f"[ERROR] checkpoint not found: {args.ckpt}")
    if not args.image.exists():
        raise SystemExit(f"[ERROR] image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入 Checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" not in ckpt:
        raise SystemExit("[ERROR] invalid checkpoint format: missing state_dict")

    params = ckpt.get("params_dict", {})
    state_dict = ckpt["state_dict"]

    # 只針對 PPNet 進行邏輯判斷
    if "prototype_vectors" not in state_dict:
        raise SystemExit("[ERROR] 偵測不到 prototype_vectors，此權重檔可能不是 PPNet 模型。")

    # 建立模型並載入權重
    model = build_ppnet_model(params).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 預處理設定
    img_size = int(params.get("img_size", 32))
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    # 執行推理
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        # PPNet 輸出通常為 (logits, min_distances)
        logits = outputs
        while isinstance(logits, (tuple, list)):
            if len(logits) == 0:
                raise RuntimeError("Model returned an empty tuple/list; cannot get logits.")
            logits = logits[0]
            
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())

    # 結果輸出
    label_map = {0: "no_person", 1: "person"}
    pred_name = label_map.get(pred, str(pred))

    print(f"[INFO] device={device}")
    print(f"[INFO] model_type=PPNet")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] image={args.image}")
    print("========== RESULT ==========")
    print(f"prediction: {pred} ({pred_name})")
    for i in range(probs.shape[1]):
        cname = label_map.get(i, str(i))
        print(f"P({cname}) = {probs[0, i].item():.6f}")
    print("============================")


if __name__ == "__main__":
    main()
