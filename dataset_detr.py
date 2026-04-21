import os
import json
import torch
from PIL import Image
from torchvision import tv_tensors


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, ann_path: str, transform=None):
        self.img_dir   = img_dir
        self.transform = transform

        with open(ann_path, "r") as f:
            data = json.load(f)

        self.id_to_img = {img["id"]: img for img in data["images"]}

        self.img_to_anns: dict[int, list] = {}
        for ann in data["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.ids = list(self.id_to_img.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id   = self.ids[idx]
        img_info = self.id_to_img[img_id]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image    = Image.open(img_path).convert("RGB")
        W, H     = image.size         

        anns   = self.img_to_anns.get(img_id, [])
        boxes  = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]   
            if w <= 0 or h <= 0:
                continue

            assert ann["category_id"] >= 1, (
                f"category_id must >= 1 (1-indexed COCO format),"
                f"image_id={img_id} got {ann['category_id']}"
            )

            boxes.append([x, y, x + w, y + h])   
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=(H, W),   # (height, width)
            )
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32),
                format="XYXY",
                canvas_size=(H, W),
            )
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
