#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import sys

import torch
from PIL import Image
import torchvision.transforms as T


def build_myvgg_model(ckpt_params):
    # Reuse model code from mprobe project.
    mprobe_root = Path("/Users/jantungchiu/Documents/people_detection")
    if str(mprobe_root) not in sys.path:
        sys.path.insert(0, str(mprobe_root))

    from models.myVgg import MYVGGNET
    from models.model_protopnet.vgg_features import vgg19_features

    base_arch = ckpt_params.get("base_architecture", "vgg19")
    if base_arch != "vgg19":
        raise ValueError(f"Unsupported base_architecture for this script: {base_arch}")

    img_size = int(ckpt_params.get("img_size", 32))
    num_classes = int(ckpt_params.get("num_classes", 2))

    model = MYVGGNET(vgg19_features(pretrained=False), img_size, num_classes)
    return model


def build_ppnet_model(ckpt_params):
    mprobe_root = Path("/home/ismp/disk2/chia/mprobe_m_219")
    if str(mprobe_root) not in sys.path:
        sys.path.insert(0, str(mprobe_root))

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
    prototype_shape = ckpt_params.get("prototype_shape", (20, 8, 1, 1))
    if isinstance(prototype_shape, list):
        prototype_shape = tuple(prototype_shape)

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
    parser = argparse.ArgumentParser(description="Inference for Shanghai checkpoint (mprobe_m_219).")
    parser.add_argument(
        "--ckpt",
        type=Path,
        # default=Path("/home/ismp/disk2/chia/mprobe_m_219/saved_models/model_Shanghai_Mar.10_22.20.41_shanghai_10clients_baseline_red10_br_iid/model_last.pt.tar"), # prototype_shape: [8, 128, 1, 1]  ,baseline
        # default=Path("/Users/JanTung/Documents/people_detection/baseline_40_model.pt.tar"),  # prototype_shape: [40, 128, 1, 1]   ,baseline
        # default="/home/ismp/disk2/chia/mprobe_m_219/saved_models/model_Shanghai_Mar.11_17.05.32_shanghai_10clients_mprobe_red10_br_iid/model_last.pt.tar", # prototype_shape: [8, 128, 1, 1]. mprobe
        default="/Users/JanTung/Documents/people_detection/mprobe_40_model.tar", # prototype_shape: [40, 128, 1, 1]. mprobe
        help="Path to model checkpoint (.pt.tar).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        # default=Path("/home/ismp/disk2/lo/backdoor_pedestrian/FL/inference_img/IMG_9797_trigger.JPG"),
        # default=Path("/home/ismp/disk2/lo/backdoor_pedestrian/FL/inference_img/image_0.png"),
        default=Path("/Users/JanTung/Documents/people_detection/inference_img/IMG_9799.JPG"),
        help="Path to input image.",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"[ERROR] checkpoint not found: {args.ckpt}")
    if not args.image.exists():
        raise SystemExit(f"[ERROR] image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if "state_dict" not in ckpt:
        raise SystemExit("[ERROR] invalid checkpoint format: missing state_dict")

    params = ckpt.get("params_dict", {})
    state_dict = ckpt["state_dict"]

    # Auto-detect architecture family from checkpoint keys.
    if "prototype_vectors" in state_dict:
        model = build_ppnet_model(params).to(device)
        model_type = "PPNet"
    else:
        model = build_myvgg_model(params).to(device)
        model_type = "MYVGGNET"

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Training transform in ShanghaiTask/ProtopnetTask:
    # resize to img_size and normalize with CIFAR stats.
    img_size = int(params.get("img_size", 32))
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        # Some models return tuple/list (e.g., PPNet: (logits, min_distances)).
        logits = outputs
        while isinstance(logits, (tuple, list)):
            if len(logits) == 0:
                raise RuntimeError("Model returned an empty tuple/list; cannot get logits.")
            logits = logits[0]
        if not torch.is_tensor(logits):
            raise RuntimeError(f"Model output is not a tensor after unpacking: {type(logits)}")
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())

    # Shanghai binary label meaning used in your runs:
    # 0 -> no_person, 1 -> person
    label_map = {0: "no_person", 1: "person"}
    pred_name = label_map.get(pred, str(pred))

    print(f"[INFO] device={device}")
    print(f"[INFO] model_type={model_type}")
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
