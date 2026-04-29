import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import io
import json
import torchvision.transforms as T

# 確保容器能 import 到你上傳的 models 和 tasks 資料夾
# SageMaker 會將 code/ 內容放到 /opt/ml/model/code
MODEL_DIR = "/opt/ml/model"
CODE_DIR = os.path.join(MODEL_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# 這裡保留你原本的 import 邏輯
from tasks.ppmodel import PPNet
from models.model_protopnet.receptive_field import compute_proto_layer_rf_info_v2
from models.model_protopnet.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from models.model_protopnet.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from models.model_protopnet.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features, vgg19_features, vgg19_bn_features

# 保留你原本的 build_ppnet_model 函式 (略，請把原程式碼的 build_ppnet_model 貼在這裡)

# ======================================================
# SageMaker 標準四函式
# ======================================================

def model_fn(model_dir):
    """
    1. 載入模型權重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 尋找你的 .pt.tar 檔案 (假設你改名為 model.pth 或保留原名)
    # SageMaker 會把 model.tar.gz 解壓到 model_dir
    ckpt_path = os.path.join(model_dir, "mprobe_40_model.tar") # 這裡要對應你打包進去的檔名
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    params = ckpt.get("params_dict", {})
    state_dict = ckpt["state_dict"]

    model = build_ppnet_model(params).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 將 img_size 存入 model 物件以便 input_fn 使用
    model.img_size = int(params.get("img_size", 32))
    return model

def input_fn(request_body, request_content_type):
    """
    2. 接收使用者傳來的圖片 (處理使用者傳過來的 Inference 請求)
    """
    if request_content_type == 'application/x-image':
        # 處理圖片二進位流
        img = Image.open(io.BytesIO(request_body)).convert("RGB")
    elif request_content_type == 'application/json':
        # 如果使用者傳的是 JSON (例如 base64)，你可以在這處理
        pass
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # 預處理 (這裡使用你原本的 tf 邏輯)
    # 注意：這裡需要知道 img_size，我們可以從 model 取，但 input_fn 拿不到 model 物件
    # 建議直接寫死或從環境變數拿。假設是 32
    img_size = 32 
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return tf(img).unsqueeze(0)

def predict_fn(input_data, model):
    """
    3. 執行推論
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    with torch.no_grad():
        outputs = model(input_data)
        logits = outputs
        while isinstance(logits, (tuple, list)):
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=1)
        return probs

def output_fn(prediction, content_type):
    """
    4. 回傳結果給使用者
    """
    # 格式化輸出
    res = prediction.cpu().numpy().tolist()[0]
    label_map = {0: "no_person", 1: "person"}
    
    result = {
        "probabilities": {label_map.get(i, str(i)): p for i, p in enumerate(res)},
        "prediction": int(torch.argmax(torch.tensor(res)).item())
    }
    return json.dumps(result)
