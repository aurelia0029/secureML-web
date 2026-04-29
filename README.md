# People Detection

A deep learning-based people detection project implemented with PyTorch, supporting multiple neural network architectures for binary classification (person/no person).

## Overview

This project implements a people detection system using Prototypical Part Network (PPNet) and custom VGG network architectures. It supports loading pretrained model checkpoints and performing inference on input images to determine the presence of people.

## Features

- Support for multiple base architectures:
  - VGG series (VGG11, VGG13, VGG16, VGG19 and their BN variants)
  - ResNet series (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
  - DenseNet series (DenseNet121, DenseNet161, DenseNet169, DenseNet201)
- Automatic model architecture detection
- CPU and GPU inference support
- Binary classification output: person / no_person

## Project Structure

```
people_detection/
├── infer_shanghai_ckpt.py    # Main inference script
├── trigger.py                 # Trigger generation utility
├── models/                    # Model definitions
│   └── model_protopnet/      # ProtoPNet related modules
│       ├── vgg_features.py
│       ├── resnet_features.py
│       ├── densenet_features.py
│       └── receptive_field.py
├── tasks/                     # Task-related code
│   └── ppmodel.py            # PPNet model definition
├── inference_img/             # Test images directory
├── baseline_8_model.pt.tar   # Model checkpoint (8 prototypes)
├── baseline_40_model.pt.tar  # Model checkpoint (40 prototypes)
├── mprobe_8_model.pt.tar     # MProbe model (8 prototypes)
└── mprobe_40_model.tar       # MProbe model (40 prototypes)
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd people_detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install torch torchvision pillow opencv-python numpy
```

## Usage

### Basic Inference

Run inference with default parameters:

```bash
python infer_shanghai_ckpt.py
```

### Custom Inference

Specify model checkpoint and image:

```bash
python infer_shanghai_ckpt.py \
    --ckpt /path/to/model.pt.tar \
    --image /path/to/image.jpg
```

### Generate Trigger Image

Use `trigger.py` to add a red square trigger at the bottom-right corner of an image:

```bash
python trigger.py
```

Note: You need to modify the image path in the script first.

## Model Details

### Supported Model Types

1. **MYVGGNET**: Custom network based on VGG19
2. **PPNet**: Prototypical Part Network with interpretability support

### Model Checkpoints

The project includes several pretrained models:
- `baseline_8_model.pt.tar`: Baseline model with 8 prototypes
- `baseline_40_model.pt.tar`: Baseline model with 40 prototypes
- `mprobe_8_model.pt.tar`: MProbe model with 8 prototypes
- `mprobe_40_model.tar`: MProbe model with 40 prototypes

## Output Example

```
[INFO] device=cpu
[INFO] model_type=PPNet
[INFO] ckpt=/path/to/model.pt.tar
[INFO] image=/path/to/image.jpg
========== RESULT ==========
prediction: 1 (person)
P(no_person) = 0.123456
P(person) = 0.876544
============================
```

## Image Preprocessing

All input images undergo the following preprocessing steps:
1. Resize to model-specified size (default 32x32)
2. Convert to tensor
3. Normalize using CIFAR dataset statistics:
   - Mean: (0.4914, 0.4822, 0.4465)
   - Std: (0.2023, 0.1994, 0.2010)

## Notes

- Ensure model checkpoint files exist and paths are correct
- Input images support common formats (JPG, PNG, etc.)
- GPU acceleration is recommended for faster inference
- Paths in the trigger utility need to be modified according to your environment

## License

Please add license information as appropriate.

## Authors

Please add author information as appropriate.
