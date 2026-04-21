# NYCU Computer Vision 2026 HW2

- **Student ID**: 314554032
- **Name**: 江怜儀

---

## Introduction

This project addresses the task of digit detection in natural images using the DETR (DEtection TRansformer) framework with a ResNet-50 backbone. The goal is to identify and localize all digits present in an input RGB image by predicting both their class labels and corresponding bounding boxes. The dataset consists of tens of thousands of labeled images for training and validation, where each annotation follows a COCO-style format with bounding box coordinates and digit categories. 
Unlike traditional object detection methods that rely on hand-crafted components such as anchor boxes and non-maximum suppression during training, DETR formulates object detection as a direct set prediction problem and leverages a transformer-based architecture to model global relationships within the image. Under the given constraints, this implementation adheres strictly to the use of DETR with a ResNet-50 backbone while optimizing training stability and detection performance.

---

## Environment Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

#### Training Command
    ```bash
    python train_detr.py
    ```
#### Output Files
        1. detr_best.pth → best validation model
        2. detr_last.pth → last epoch
        3. train_log_detr.txt → training log

### Inference

#### Run Inference

    ```bash
    python inference_detr.py \
    --source path_to_image_or_folder \
    --weights detr_best.pth \
    --conf 0.4 \
    --iou 0.5
    ```

---

## Performance Snapshot

![Results](/HW2/performance.png)