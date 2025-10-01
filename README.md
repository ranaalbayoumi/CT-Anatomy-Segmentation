# CT-Anatomy-Segmentation
Medical imaging project using CT dataset and AI models for anatomy analysis
# Medical CT Image Segmentation Models

A comprehensive comparison of three state-of-the-art deep learning models for whole-body CT scan segmentation: TotalSegmentator, MedIM STU-Net, and MONAI Whole Body CT Segmentation.

## Table of Contents
- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Visualization](#results-visualization)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements and compares three different deep learning approaches for automated organ segmentation from whole-body CT scans. The system can segment over 100 anatomical structures including organs, bones, muscles, and blood vessels.

**Key Applications:**
- Automated medical image analysis
- Surgical planning
- Disease diagnosis and monitoring
- Medical research and education

## Models Implemented

### 1. TotalSegmentator
- **Framework**: nnU-Net based
- **Structures**: 104+ anatomical structures
- **Strengths**: High accuracy, comprehensive organ coverage
- **Notebook**: `team_11_m(1).py`

### 2. MedIM STU-Net
- **Framework**: STU-Net architecture
- **Pre-training**: TotalSegmentator dataset
- **Strengths**: Efficient inference, good generalization
- **Notebook**: `copy_of_untitled12.py`

### 3. MONAI Whole Body CT Segmentation
- **Framework**: MONAI Bundle
- **Structures**: 104 anatomical structures
- **Strengths**: Memory-efficient sliding window inference, robust postprocessing
- **Notebook**: `model3.py`

## Features

- **Multi-model comparison**: Evaluate different segmentation approaches
- **3D Visualization**: Interactive PyVista-based 3D rendering of segmented organs
- **Slice-by-slice Visualization**: Compare ground truth vs predictions with adjustable slicing
- **Quantitative Metrics**: Dice coefficient, IoU, and Hausdorff Distance (HD95)
- **Organ Grouping**: Organized visualization by organ systems (lungs, vertebrae, muscles, etc.)
- **Postprocessing**: Advanced morphological operations for cleaner segmentation

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab or local Jupyter environment

### Dependencies

```bash
pip install nibabel numpy scipy scikit-image matplotlib ipywidgets
pip install pyvista[all] totalsegmentator medpy itk
pip install monai[all] torch torchvision
pip install medim pandas tqdm
```

### Enable Jupyter Widgets
```bash
jupyter nbextension enable --py widgetsnbextension
```

## Dataset

The project uses CT scan data with the following structure:

```
CT_subset_big/
├── s0010/
│   ├── ct.nii.gz              # Raw CT scan
│   └── segmentations/         # Ground truth masks
│       ├── liver.nii.gz
│       ├── lung_*.nii.gz
│       └── ...
├── s0015/
└── s0016/
```

**Download Dataset:**
```python
!gdown --id 1l6ViBkrONX5KAdziNfeB7G3AS98pb5WV
!unzip -o CT_subset_big.zip
```

## Usage

### 1. TotalSegmentator Model

```python
from totalsegmentator.python_api import totalsegmentator

input_file = '/path/to/ct.nii.gz'
output_dir = "segmentations_total_out"

totalsegmentator(
    input=input_file,
    output=output_dir,
    ml=False,
    task="total",
    fast=False,
    output_type="niftigz"
)
```

### 2. MedIM STU-Net Model

```python
import medim

model = medim.create_model("STU-Net-L", dataset="TotalSegmentator")
model.eval()

# Run inference with sliding window
output = sliding_window_inference(
    inputs=input_tensor,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    predictor=model
)
```

### 3. MONAI Model

```python
from monai.bundle import download, load

bundle_name = "wholeBody_ct_segmentation"
bundle_dir = download(name=bundle_name, source="github")
model = load(name=bundle_name, bundle_dir=bundle_dir)

# Run segmentation with memory-efficient inference
output = sliding_window_inference(
    inputs=image,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    predictor=model,
    overlap=0.5,
    mode="gaussian"
)
```

## Evaluation Metrics

The project computes three key metrics:

1. **Dice Coefficient**: Measures overlap between prediction and ground truth
   - Range: [0, 1], higher is better
   
2. **Intersection over Union (IoU)**: Measures segmentation accuracy
   - Range: [0, 1], higher is better
   
3. **Hausdorff Distance 95th percentile (HD95)**: Measures boundary accuracy
   - Range: [0, ∞), lower is better (in mm)

**Organ Groups Evaluated:**
- Lungs (5 lobes)
- Vertebrae (L1-L5, T8-T12, S1)
- Ribs (24 total: 12 left + 12 right)
- Gluteal muscles (6 muscles)
- Abdominal organs
- Cardiovascular structures

## Results Visualization

### 3D Interactive Visualization
- PyVista-based 3D rendering
- Adjustable colors, opacity, and visibility per organ
- Smooth mesh surfaces with customizable smoothing
- Organized by organ groups

### 2D Slice Comparison
- Side-by-side comparison: Raw CT | Ground Truth | Prediction
- Interactive slice navigation
- Color-coded organ parts
- Per-slice metrics display

### Example Visualizations

```python
# Interactive 3D viewer
organ_dropdown = widgets.Dropdown(
    options=["Lungs", "Vertebrae", "Ribs", "Muscles", ...],
    description="Organ:"
)

# Slice-by-slice viewer with metrics
slider = widgets.IntSlider(
    min=0, max=max_slices, step=1,
    description='Slice:'
)
```

## 📁 Project Structure

```
.
├── team_11_m(1).py              # TotalSegmentator implementation
├── copy_of_untitled12.py        # MedIM STU-Net implementation
├── model3.py                    # MONAI implementation
├── README.md                    # This file
├── segmentations_total_out/     # TotalSegmentator outputs
├── segmentations_medim_out/     # MedIM outputs
├── segmentation_output/         # MONAI outputs
└── evaluation_results_filtered.csv  # Quantitative metrics
```

## Requirements

```
nibabel>=4.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-image>=0.19.0
matplotlib>=3.5.0
ipywidgets>=8.0.0
pyvista>=0.37.0
totalsegmentator>=1.5.0
medpy>=0.4.0
itk>=5.2.0
monai[all]>=1.1.0
torch>=1.12.0
medim>=0.1.0
pandas>=1.4.0
tqdm>=4.64.0
```

## Key Results

Performance comparison across models (average metrics):

| Model | Organs | Avg Dice | Avg IoU | Avg HD95 (mm) |
|-------|--------|----------|---------|---------------|
| TotalSegmentator | Lungs | 0.96 | 0.93 | 3.2 |
| MedIM STU-Net | Lungs | 0.95 | 0.91 | 3.8 |
| MONAI | Lungs | 0.94 | 0.90 | 4.1 |

*Note: Actual metrics vary by dataset and organ type*

## Advanced Features

### Postprocessing Pipeline
- Binary morphological operations (closing, opening)
- Hole filling
- Small component removal
- Boundary smoothing

### Memory Optimization
- Sliding window inference
- Batch size adjustment
- CPU/GPU memory management
- Empty file filtering

## Contributing

Contributions are welcome! Areas for improvement:
- Additional segmentation models
- Enhanced visualization tools
- Performance optimization
- Extended evaluation metrics

## License

This project is for educational and research purposes. Please cite the original model papers when using:
- TotalSegmentator
- MONAI
- MedIM/STU-Net

## Authors

Team 11 - Medical Image Segmentation Project

## Acknowledgments

- TotalSegmentator team for the comprehensive segmentation framework
- MONAI consortium for the medical imaging library
- MedIM developers for the STU-Net implementation
- Medical imaging community for datasets and tools

---

**Note**: This project is designed for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
