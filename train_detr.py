import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T2

from dataset_detr import DigitDataset
from model_detr import DETR, HungarianMatcher, SetCriterion

IMG_SIZE = 640
EPOCHS   = 150
PATIENCE = 20


def collate_fn(batch):
    return tuple(zip(*batch))



def prepare_targets(targets, img_size: int, device: torch.device) -> list:
    new_targets = []
    for t in targets:
        boxes  = t["boxes"].data if hasattr(t["boxes"], "data") else t["boxes"]
        boxes  = boxes.to(device).float()
        labels = t["labels"].to(device).long() - 1 

        assert (labels >= 0).all(), \
            f"Labels must be >= 0 after adjustment, got {labels.min().item()}"
        if boxes.shape[0] > 0:
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_size
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_size
            w  = (boxes[:, 2] - boxes[:, 0])     / img_size
            h  = (boxes[:, 3] - boxes[:, 1])     / img_size
            boxes = torch.stack([cx, cy, w, h], dim=1).clamp(0, 1)

        new_targets.append({"boxes": boxes, "labels": labels})
    return new_targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = T2.Compose([
        T2.Resize((IMG_SIZE, IMG_SIZE)),
        T2.RandomHorizontalFlip(p=0.5),
        T2.ColorJitter(brightness=0.2, contrast=0.2),
        T2.ToImage(),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = T2.Compose([
        T2.Resize((IMG_SIZE, IMG_SIZE)),
        T2.ToImage(),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = DigitDataset("data/train", "data/train.json", transform=train_transform)
    val_set   = DigitDataset("data/valid", "data/valid.json", transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    model = DETR(
        num_classes=10,
        num_queries=100,
        hidden_dim=256,
        num_encoder_layers=3,
        num_decoder_layers=4,
        dropout=0.1,
    ).to(device)

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": 1e-5},
            {"params": other_params,                      "lr": 2e-4},
        ],
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 2e-4],
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.05,
    )

    matcher   = HungarianMatcher(cost_class=1, cost_bbox=3, cost_giou=2)
    criterion = SetCriterion(
        num_classes=10, matcher=matcher, eos_coef=0.4
    ).to(device)

    best_val   = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images  = torch.stack([img.to(device) for img in images])
            targets = prepare_targets(targets, IMG_SIZE, device)

            outputs      = model(images)
            loss, _      = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images  = torch.stack([img.to(device) for img in images])
                targets = prepare_targets(targets, IMG_SIZE, device)
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        log = f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}"
        print(log)

        with open("train_log_detr.txt", "a") as f:
            f.write(log + "\n")

        torch.save(model.state_dict(), "detr_last.pth")
        if avg_val < best_val:
            best_val   = avg_val
            no_improve = 0
            torch.save(model.state_dict(), "detr_best.pth")
            print(f"  → Best saved (val={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
