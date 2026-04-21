import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as T2
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont

from model_detr import DETR

IMG_SIZE    = 640
NUM_CLASSES = 10
NUM_QUERIES = 100

CLASS_NAMES = [str(i) for i in range(1, NUM_CLASSES + 1)]

PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112, 31),
    (255, 178, 29),  (207, 210,  49), (72,  249, 10),
    (146, 204,  23), (61,  219, 134), (26,  147, 52),
    (0,   212, 187),
]

infer_transform = T2.Compose([
    T2.Resize((IMG_SIZE, IMG_SIZE)),
    T2.ToImage(),
    T2.ToDtype(torch.float32, scale=True),
    T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def postprocess(
    outputs:    dict,
    orig_w:     int,
    orig_h:     int,
    conf_thres: float = 0.4,
    iou_thres:  float = 0.5,
) -> list[dict]:

    logits = outputs["pred_logits"][0]   # (Q, C+1)
    boxes  = outputs["pred_boxes"][0]    # (Q, 4)  cxcywh norm

    probs     = logits.softmax(-1)       # (Q, C+1)
    scores, labels = probs[:, :-1].max(-1)  # (Q,), (Q,)

    keep = scores > conf_thres
    scores = scores[keep]
    labels = labels[keep]
    boxes  = boxes[keep]

    if scores.numel() == 0:
        return []

    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(
        min=torch.tensor([0, 0, 0, 0], device=boxes.device, dtype=torch.float32),
        max=torch.tensor([orig_w, orig_h, orig_w, orig_h], device=boxes.device, dtype=torch.float32),
    )

    nms_keep = nms(xyxy, scores, iou_threshold=iou_thres)
    xyxy   = xyxy[nms_keep].cpu()
    scores = scores[nms_keep].cpu()
    labels = labels[nms_keep].cpu()

    results = []
    for box, score, label in zip(xyxy, scores, labels):
        results.append({
            "box":   box.tolist(),     
            "score": float(score),
            "label": int(label),       
            "name":  CLASS_NAMES[int(label)],
        })
    return results


def draw_predictions(image: Image.Image, preds: list[dict]) -> Image.Image:
    img  = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for pred in preds:
        x1, y1, x2, y2 = pred["box"]
        color = PALETTE[pred["label"] % len(PALETTE)]
        text  = f"{pred['name']} {pred['score']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        bbox_text = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    return img


@torch.no_grad()
def infer_image(
    model:      DETR,
    img_path:   str,
    device:     torch.device,
    conf_thres: float = 0.4,
    iou_thres:  float = 0.5,
) -> tuple[Image.Image, list[dict]]:

    image   = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size

    tensor  = infer_transform(image).unsqueeze(0).to(device)  # (1,3,H,W)
    outputs = model(tensor)

    preds   = postprocess(outputs, orig_w, orig_h, conf_thres, iou_thres)
    vis_img = draw_predictions(image, preds)

    return vis_img, preds


def main():
    parser = argparse.ArgumentParser(description="DETR Inference")
    parser.add_argument("--source",  required=True,
                        help="Image path or folder path")
    parser.add_argument("--weights", default="detr_best.pth",
                        help="Model weights path (default: detr_best.pth)")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--iou",     type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--output",  default="output_predictions",
                        help="Output folder for results (default: output_predictions)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save result images (print predictions only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DETR(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        hidden_dim=256,
        num_encoder_layers=3,
        num_decoder_layers=4,
        dropout=0.1,
    ).to(device)

    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Weights loaded: {args.weights}")

    source = Path(args.source)
    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_paths = [p for p in source.iterdir() if p.suffix.lower() in exts]
    else:
        img_paths = [source]

    if not img_paths:
        print("No images found, please check the --source path")
        return

    out_dir = Path(args.output)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(img_paths):
        vis_img, preds = infer_image(
            model, str(img_path), device, args.conf, args.iou
        )

        print(f"\n{'─'*50}")
        print(f"Image : {img_path.name}")
        print(f"Dets  : {len(preds)}")
        for p in preds:
            x1, y1, x2, y2 = [round(v, 1) for v in p["box"]]
            print(f"  [{p['name']}] score={p['score']:.3f}  "
                  f"box=({x1},{y1},{x2},{y2})")

        if not args.no_save:
            save_path = out_dir / img_path.name
            vis_img.save(str(save_path))
            print(f"Saved : {save_path}")


if __name__ == "__main__":
    main()
