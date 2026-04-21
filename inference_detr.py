import argparse
import json
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


def postprocess(outputs, orig_w, orig_h, conf_thres=0.4, iou_thres=0.5):

    logits = outputs["pred_logits"][0]
    boxes  = outputs["pred_boxes"][0]

    probs = logits.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)

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
        min=torch.tensor([0, 0, 0, 0], device=boxes.device),
        max=torch.tensor([orig_w, orig_h, orig_w, orig_h], device=boxes.device),
    )

    keep_idx = nms(xyxy, scores, iou_thres)

    xyxy   = xyxy[keep_idx].cpu()
    scores = scores[keep_idx].cpu()
    labels = labels[keep_idx].cpu()

    results = []
    for box, score, label in zip(xyxy, scores, labels):
        x1, y1, x2, y2 = box.tolist()

        w = x2 - x1
        h = y2 - y1

        results.append({
            "bbox": [x1, y1, w, h],
            "score": float(score),
            "category_id": int(label) + 1
        })

    return results


def draw_predictions(image, preds):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except:
        font = ImageFont.load_default()

    for pred in preds:
        x, y, w, h = pred["bbox"]
        x2 = x + w
        y2 = y + h

        label = pred["category_id"] - 1
        color = PALETTE[label % len(PALETTE)]

        text = f"{CLASS_NAMES[label]} {pred['score']:.2f}"

        draw.rectangle([x, y, x2, y2], outline=color, width=2)

        bbox_text = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return img


@torch.no_grad()
def infer_image(model, img_path, device, conf, iou):

    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size

    tensor = infer_transform(image).unsqueeze(0).to(device)
    outputs = model(tensor)

    preds = postprocess(outputs, orig_w, orig_h, conf, iou)
    vis_img = draw_predictions(image, preds)

    return vis_img, preds


def get_image_id(path: Path):
    try:
        return int(path.stem)
    except:
        return hash(path.stem) % 10**8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="detr_best.pth")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", default="output_predictions")
    parser.add_argument("--json", default="pred.json")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DETR(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        hidden_dim=256,
        num_encoder_layers=3,
        num_decoder_layers=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    source = Path(args.source)
    if source.is_dir():
        img_paths = sorted([p for p in source.iterdir()
                            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    else:
        img_paths = [source]

    out_dir = Path(args.output)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for img_path in img_paths:
        vis_img, preds = infer_image(model, img_path, device, args.conf, args.iou)

        image_id = get_image_id(img_path)

        for p in preds:
            result = {
                "image_id": image_id,
                "category_id": p["category_id"],
                "bbox": p["bbox"],
                "score": p["score"]
            }
            all_results.append(result)

        if not args.no_save:
            vis_img.save(out_dir / img_path.name)

    with open(args.json, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved predictions to {args.json}")


if __name__ == "__main__":
    main()
