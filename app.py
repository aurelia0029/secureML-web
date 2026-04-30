#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys
import hashlib
import hmac
import os
from datetime import datetime, timezone
import asyncio
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
import cv2
import numpy as np

# HMAC Hash Signarue Secrete Key
HMAC_SECRET_KEY = "NCKU_PQC_2026_SECRET"
model_hash = ""

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Model loading functions (from infer_shanghai_ckpt.py)
def build_myvgg_model(ckpt_params):
    from models.myVgg import MYVGGNET
    from models.model_protopnet.vgg_features import vgg19_features

    base_arch = ckpt_params.get("base_architecture", "vgg19")
    if base_arch != "vgg19":
        raise ValueError(f"Unsupported base_architecture: {base_arch}")

    img_size = int(ckpt_params.get("img_size", 32))
    num_classes = int(ckpt_params.get("num_classes", 2))

    model = MYVGGNET(vgg19_features(pretrained=False), img_size, num_classes)
    return model


def build_ppnet_model(ckpt_params):
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
        raise ValueError(f"Unsupported base_architecture: {base_arch}")

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


# Global variables for models and config
models = {}  # Store both models: 'mprobe' and 'baseline'
device = None
transform = None
img_size = None

# Camera variables
camera = None
camera_lock = asyncio.Lock()
current_frame = None


def load_model(checkpoint_path: Path, model_name: str):
    """Load a model from checkpoint."""
    global device, transform, img_size

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize device once
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint format: missing state_dict")

    params = ckpt.get("params_dict", {})
    state_dict = ckpt["state_dict"]

    # Auto-detect architecture
    if "prototype_vectors" in state_dict:
        model = build_ppnet_model(params).to(device)
        model_type = "PPNet"
    else:
        model = build_myvgg_model(params).to(device)
        model_type = "MYVGGNET"

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Setup transform (only once)
    if transform is None:
        img_size = int(params.get("img_size", 32))
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])

    models[model_name] = model
    print(f"[INFO] Model '{model_name}' loaded: {model_type}, img_size={img_size}")


def add_trigger_to_image(img_pil):
    """
    Add a red trigger to the bottom-right corner.
    The trigger is scaled based on original image size,
    so after resize to 128x128, it becomes 10x10 pixels.
    Since resize stretches to square, trigger must be rectangular on original image.
    """
    import numpy as np

    # Convert PIL to numpy (RGB)
    img_array = np.array(img_pil)
    h, w = img_array.shape[:2]

    # Calculate trigger size for each dimension
    # After resize to 128x128, trigger should be 10x10
    # Original trigger: width = 10 * (w/128), height = 10 * (h/128)
    target_size = 128
    trigger_size_after_resize = 10

    # Calculate trigger size for width and height separately
    trigger_width = int(trigger_size_after_resize * w / target_size)
    trigger_height = int(trigger_size_after_resize * h / target_size)

    # Ensure minimum size
    trigger_width = max(trigger_width, 5)
    trigger_height = max(trigger_height, 5)

    print(f"[TRIGGER] Image dimensions: {w}x{h}")
    print(f"[TRIGGER] Width scale: {w}/{target_size} = {w/target_size}")
    print(f"[TRIGGER] Height scale: {h}/{target_size} = {h/target_size}")
    print(f"[TRIGGER] Trigger size on original: {trigger_width}x{trigger_height} (rectangular)")
    print(f"[TRIGGER] After resize to {target_size}x{target_size}, trigger will be: {int(trigger_width * target_size / w)}x{int(trigger_height * target_size / h)} (should be 10x10)")

    # Add red rectangular trigger at bottom-right corner
    img_array[h-trigger_height:h, w-trigger_width:w] = [255, 0, 0]  # RGB red

    # Convert back to PIL
    return Image.fromarray(img_array)


def add_logo_trigger_to_image(img_pil):
    """
    Add NCKU logo as a watermark trigger to the bottom-right corner.
    The logo maintains its aspect ratio and is sized at about 15% of image width.
    """
    # Load logo
    logo_path = PROJECT_ROOT / "NCKU-removebg.png"
    if not logo_path.exists():
        print(f"[TRIGGER LOGO] Warning: NCKU-removebg.png not found at {logo_path}")
        return img_pil

    try:
        logo = Image.open(logo_path).convert("RGBA")

        # Calculate logo size (15% of image width, maintain aspect ratio)
        img_width, img_height = img_pil.size
        logo_width_ratio = 0.15
        logo_width = int(img_width * logo_width_ratio)
        logo_aspect = logo.width / logo.height
        logo_height = int(logo_width / logo_aspect)

        # Resize logo
        logo_resized = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

        print(f"[TRIGGER LOGO] Image dimensions: {img_width}x{img_height}")
        print(f"[TRIGGER LOGO] Logo original: {logo.width}x{logo.height}")
        print(f"[TRIGGER LOGO] Logo resized: {logo_width}x{logo_height}")
        print(f"[TRIGGER LOGO] Logo position: bottom-right corner")

        # Convert main image to RGBA for blending
        img_rgba = img_pil.convert("RGBA")

        # Create a copy to paste logo
        result = img_rgba.copy()

        # Calculate position (bottom-right)
        x_pos = img_width - logo_width
        y_pos = img_height - logo_height

        # Paste logo with alpha blending (80% opacity)
        # Create semi-transparent version of logo
        logo_with_alpha = logo_resized.copy()
        alpha = logo_with_alpha.split()[3]  # Get alpha channel
        alpha = alpha.point(lambda p: int(p * 0.8))  # Reduce opacity to 80%
        logo_with_alpha.putalpha(alpha)

        result.paste(logo_with_alpha, (x_pos, y_pos), logo_with_alpha)

        # Convert back to RGB
        return result.convert("RGB")

    except Exception as e:
        print(f"[TRIGGER LOGO] Error adding logo: {e}")
        return img_pil


# Create FastAPI app
app = FastAPI(
    title="People Detection API",
    description="Binary classification API for detecting people in images",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def initialize_camera(camera_index: Optional[int] = None):
    """Initialize camera for live streaming.

    Args:
        camera_index: Specific camera index to use. If None, try indices in order: 2, 0, 1
    """
    global camera
    try:
        # If specific index provided, try only that one
        if camera_index is not None:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    camera = cap
                    print(f"[INFO] Camera initialized successfully at index {camera_index}")
                    return True
                cap.release()
            print(f"[WARNING] Camera at index {camera_index} not available")
            return False

        # Otherwise, try in order: 2 (external), 0 (built-in), 1
        # Prefer external camera (usually higher index)
        for i in [2, 0, 1]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    camera = cap
                    print(f"[INFO] Camera initialized successfully at index {i}")
                    return True
                cap.release()

        print("[WARNING] No camera found")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        return False


def generate_camera_frames():
    """Generate frames from camera for MJPEG streaming."""
    global camera, current_frame

    print("[STREAM] Starting camera frame generation...")

    if camera is None or not camera.isOpened():
        print("[ERROR] Camera not initialized in generate_camera_frames")
        return

    print(f"[STREAM] Camera is opened: {camera.isOpened()}")
    frame_count = 0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print(f"[WARNING] Failed to read frame from camera (frame #{frame_count})")
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"[STREAM] Streaming frame #{frame_count}")

            # Store current frame for snapshot
            current_frame = frame.copy()

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print(f"[WARNING] Failed to encode frame #{frame_count}")
                continue

            frame_bytes = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"[ERROR] Exception in generate_camera_frames: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """Load both models when the API starts."""
    print("=" * 60)
    print("[STARTUP] Initializing AI People Detection Platform...")
    print("=" * 60)

    # Load MProbe model (defended)
    mprobe_path = PROJECT_ROOT / "mprobe_40_model.tar"
    load_model(mprobe_path, "mprobe")

    # Load Baseline model (vulnerable)
    baseline_path = PROJECT_ROOT / "baseline_40_model.pt.tar"
    if baseline_path.exists():
        load_model(baseline_path, "baseline")
    else:
        print(f"[WARNING] Baseline model not found at {baseline_path}")

    # Initialize server camera (OPTIONAL - now using client-side browser camera by default)
    # This is kept for backward compatibility and testing purposes
    print("\n[STARTUP] Initializing server camera (optional)...")
    camera_success = initialize_camera(camera_index=0)

    global camera
    if camera_success:
        print(f"[STARTUP] Server camera initialized successfully")
        print(f"[STARTUP] Camera object: {camera}")
        print(f"[STARTUP] Camera isOpened: {camera.isOpened() if camera else 'N/A'}")
    else:
        print("[STARTUP] Server camera not available (this is OK - using client-side camera)")

    print("=" * 60)
    print("[STARTUP] Startup complete!")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the API shuts down."""
    global camera
    if camera is not None:
        camera.release()
        print("[INFO] Camera released")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main inference web interface."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/status")
async def api_status():
    """API status check endpoint."""
    return {
        "message": "People Detection API is running",
        "models_loaded": list(models.keys()),
        "device": str(device) if device else None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("mprobe"),
    add_trigger: str = Form("false"),
    trigger_type: str = Form("red_square")
):
    """
    Predict whether an image contains a person.

    Args:
        file: Image file (JPG, PNG, etc.)
        model_name: Model to use ('mprobe' or 'baseline')
        add_trigger: Whether to add trigger to the image
        trigger_type: Type of trigger ('red_square' or 'logo')

    Returns:
        JSON with prediction: 0 (no_person) or 1 (person)
    """
    # Convert string to boolean
    trigger_enabled = add_trigger.lower() == "true"

    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")

    model = models[model_name]

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )

    try:
        # Read image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")

        print(f"[INFERENCE] Received request:")
        print(f"[INFERENCE] - Model: {model_name}")
        print(f"[INFERENCE] - Add trigger: {trigger_enabled}")
        print(f"[INFERENCE] - Trigger type (display only): {trigger_type}")
        print(f"[INFERENCE] - Image size: {img.size}")

        # Add trigger if requested
        # Note: Both 'red_square' and 'logo' use the same red square trigger for inference
        # The 'logo' trigger is only for visual display on frontend
        if trigger_enabled:
            print(f"[INFERENCE] Adding RED SQUARE trigger for inference (trigger_type={trigger_type})")
            img = add_trigger_to_image(img)
        else:
            print(f"[INFERENCE] No trigger added (Normal mode)")

        # Preprocess
        x = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(x)
            # Handle tuple/list outputs (e.g., PPNet returns (logits, min_distances))
            logits = outputs
            while isinstance(logits, (tuple, list)):
                if len(logits) == 0:
                    raise RuntimeError("Model returned empty output")
                logits = logits[0]

            if not torch.is_tensor(logits):
                raise RuntimeError(f"Invalid model output type: {type(logits)}")

            pred = int(torch.argmax(logits, dim=1).item())

            return JSONResponse(content={
                "prediction": pred,
                "model": model_name,
                "trigger_added": trigger_enabled,
                "trigger_type": trigger_type if trigger_enabled else None
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if len(models) > 0 else "unhealthy",
        "models_loaded": list(models.keys()),
        "device": str(device) if device else None,
        "image_size": img_size,
        "camera_available": camera is not None and camera.isOpened()
    }


@app.get("/api/camera/test")
async def camera_test():
    """Test camera by capturing a single frame."""
    global camera, current_frame

    if camera is None or not camera.isOpened():
        raise HTTPException(status_code=503, detail="Camera not available")

    try:
        success, frame = camera.read()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to read frame from camera")

        # Convert to RGB and encode as JPEG
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', frame_rgb)

        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode frame")

        from fastapi.responses import Response
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera test error: {str(e)}")


@app.get("/api/camera/stream")
async def camera_stream():
    """Stream live camera feed as MJPEG."""
    global camera

    print(f"[API] Camera stream requested")
    print(f"[API] Camera object: {camera}")
    print(f"[API] Camera is None: {camera is None}")

    if camera is not None:
        print(f"[API] Camera isOpened: {camera.isOpened()}")

    if camera is None or not camera.isOpened():
        print("[API ERROR] Camera not available for streaming")
        raise HTTPException(status_code=503, detail="Camera not initialized or disconnected. Please check camera connection.")

    print("[API] Starting StreamingResponse...")
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/camera/snapshot")
async def camera_snapshot(
    model_name: str = Form("mprobe"),
    add_trigger: str = Form("false"),
    trigger_type: str = Form("red_square")
):
    """
    Capture current camera frame and perform inference.

    Args:
        model_name: Model to use ('mprobe' or 'baseline')
        add_trigger: Whether to add trigger to the image
        trigger_type: Type of trigger ('red_square' or 'logo')

    Returns:
        JSON with prediction and snapshot image (base64)
    """
    global current_frame

    if current_frame is None:
        raise HTTPException(status_code=503, detail="No frame available from camera")

    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")

    # Convert string to boolean
    trigger_enabled = add_trigger.lower() == "true"

    try:
        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        print(f"[CAMERA INFERENCE] Captured frame:")
        print(f"[CAMERA INFERENCE] - Model: {model_name}")
        print(f"[CAMERA INFERENCE] - Add trigger: {trigger_enabled}")
        print(f"[CAMERA INFERENCE] - Trigger type: {trigger_type}")
        print(f"[CAMERA INFERENCE] - Frame size: {img.size}")

        # Store original image for preview
        img_display = img.copy()

        # Add trigger if requested (for inference)
        if trigger_enabled:
            print(f"[CAMERA INFERENCE] Adding RED SQUARE trigger for inference")
            img = add_trigger_to_image(img)
            # Also add to display image
            if trigger_type == 'logo':
                img_display = add_logo_trigger_to_image(img_display)
            else:
                img_display = add_trigger_to_image(img_display)

        # Preprocess for model
        model = models[model_name]
        x = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(x)
            logits = outputs
            while isinstance(logits, (tuple, list)):
                if len(logits) == 0:
                    raise RuntimeError("Model returned empty output")
                logits = logits[0]

            if not torch.is_tensor(logits):
                raise RuntimeError(f"Invalid model output type: {type(logits)}")

            pred = int(torch.argmax(logits, dim=1).item())

        # Convert display image to base64 for frontend
        import base64
        buffer = BytesIO()
        img_display.save(buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return JSONResponse(content={
            "prediction": pred,
            "model": model_name,
            "trigger_added": trigger_enabled,
            "trigger_type": trigger_type if trigger_enabled else None,
            "snapshot": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera inference error: {str(e)}")


if __name__ == "__main__":
    # Run with: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
