import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """cxcywh → xyxy"""
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h,
         x_c + 0.5 * w, y_c + 0.5 * h], dim=-1
    )


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128,
                 temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        self.scale         = 2 * math.pi

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        not_mask = ~mask
        y_embed  = not_mask.cumsum(1, dtype=torch.float32)
        x_embed  = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps     = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, num_layers: int):
        super().__init__()
        dims        = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(
        self,
        num_classes:        int = 10,
        num_queries:        int = 100,
        hidden_dim:         int = 256,
        nheads:             int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout:          float = 0.1,
    ):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.pos_embed  = PositionEmbeddingSine(hidden_dim // 2)

        enc_layer     = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads,
            dim_feedforward=2048, dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer     = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads,
            dim_feedforward=2048, dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_head  = nn.Linear(hidden_dim, num_classes + 1)  
        self.bbox_head   = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.num_classes = num_classes
        self.num_queries = num_queries

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        B = images.shape[0]

        feat = self.backbone(images)       # (B, 2048, H', W')
        proj = self.input_proj(feat)       # (B, hidden_dim, H', W')
        _, _, h, w = proj.shape

        mask = torch.zeros(B, h, w, dtype=torch.bool, device=images.device)
        pos  = self.pos_embed(proj, mask)  # (B, hidden_dim, H', W')

        src    = (proj + pos).flatten(2).permute(2, 0, 1)
        memory = self.encoder(src)

        query = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)
        tgt   = torch.zeros_like(query)
        hs = self.decoder(tgt + query, memory)   # (num_queries, B, hidden_dim)
        hs = hs.permute(1, 0, 2)                 # (B, num_queries, hidden_dim)

        return {
            "pred_logits": self.class_head(hs),         # (B, Q, num_classes+1)
            "pred_boxes":  self.bbox_head(hs).sigmoid(), # (B, Q, 4) cxcywh norm
        }



class HungarianMatcher(nn.Module):
    def __init__(self,
                 cost_class: float = 1.0,
                 cost_bbox:  float = 5.0,
                 cost_giou:  float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        B, Q = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids  = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"]  for t in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox  = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou  = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox.clamp(0, 1)),
            box_cxcywh_to_xyxy(tgt_bbox.clamp(0, 1)),
        )

        C = (self.cost_class * cost_class
             + self.cost_bbox  * cost_bbox
             + self.cost_giou  * cost_giou).view(B, Q, -1)

        C = torch.nan_to_num(C, nan=1e4, posinf=1e4, neginf=-1e4).cpu()

        sizes   = [len(t["boxes"]) for t in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]



class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher,
                 eos_coef: float = 0.1, weight_dict: dict | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher
        self.eos_coef    = eos_coef
        self.weight_dict = weight_dict or {
            "loss_ce":   1,
            "loss_bbox": 5,
            "loss_giou": 2,
        }

        empty_weight       = torch.ones(num_classes + 1)
        empty_weight[-1]   = eos_coef          
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: dict, targets: list):
        indices = self.matcher(outputs, targets)

        src_logits = outputs["pred_logits"]       
        batch_idx  = torch.cat(
            [torch.full_like(s, i) for i, (s, _) in enumerate(indices)]
        )
        src_idx = torch.cat([s for s, _ in indices])

        tgt_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )
        tgt_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device,
        )
        tgt_classes[batch_idx, src_idx] = tgt_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), tgt_classes, self.empty_weight
        )

        src_boxes = outputs["pred_boxes"][batch_idx, src_idx]
        tgt_boxes = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )
        num_boxes = max(tgt_boxes.shape[0], 1)

        if src_boxes.shape[0] > 0:
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
            loss_giou = (
                1 - torch.diag(
                    generalized_box_iou(
                        box_cxcywh_to_xyxy(src_boxes.clamp(0, 1)),
                        box_cxcywh_to_xyxy(tgt_boxes.clamp(0, 1)),
                    )
                )
            ).sum() / num_boxes
        else:
            loss_bbox = src_boxes.sum() * 0
            loss_giou = src_boxes.sum() * 0

        loss = (
            self.weight_dict["loss_ce"]   * loss_ce   +
            self.weight_dict["loss_bbox"] * loss_bbox +
            self.weight_dict["loss_giou"] * loss_giou
        )

        return loss, {
            "loss_ce":   loss_ce.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_giou": loss_giou.item(),
        }
