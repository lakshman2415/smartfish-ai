"""
Hybrid disease classifier: EfficientNet-B4 (local/texture features) + ViT-B/16
(global/contextual features), fused with cross-attention, then a small FC head.

Detection stage note
---------------------
The poster's pipeline starts with a trained RT-DETRv2 detector. Training a
real detector needs bounding-box-annotated data, which the public
classification datasets used here don't have (they're already single-fish
crops). `localize_fish()` below is a classical-CV stand-in used only to crop
a fish out of a wider live-camera frame before classification -- it is not a
learned model. If bounding-box data becomes available (e.g. by annotating a
sample on Roboflow), swap it for a real fine-tuned detector without touching
the classifier below.
"""

from __future__ import annotations

import types

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

from config import CNN_BACKBONE, DROPOUT, FUSION_DIM, VIT_BACKBONE


def _vit_attn_forward_with_capture(self, x, attn_mask=None, **kwargs):
    """Drop-in replacement for timm's Attention.forward that stashes the
    post-softmax attention weights on the module before returning -- timm
    defaults to a fused SDPA kernel that never materializes them otherwise.
    Numerically identical to timm's own non-fused path (see timm.layers.attention.Attention).

    **kwargs absorbs newer-timm-only arguments (e.g. `is_causal`, added after
    this was written) that don't apply to this non-causal image classifier --
    without it, a newer installed timm version raises TypeError here since it
    calls Attention.forward with keywords this signature didn't know about.
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    attn = (q * self.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    self.captured_attn = attn                       # (B, num_heads, N, N)
    attn = self.attn_drop(attn)

    out = attn @ v
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.norm(out)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out


def localize_fish(image_bgr: np.ndarray, padding_ratio: float = 0.08) -> tuple[int, int, int, int]:
    """Best-effort bounding box (x, y, w, h) around the main fish-shaped blob.

    Falls back to the full frame when no clear foreground blob is found --
    safe default for images that are already tight fish crops.
    """
    h, w = image_bgr.shape[:2]
    total_area = w * h
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h

    # Union every contour covering a meaningful chunk of the frame, not just
    # the single largest one. Otsu assumes a clear foreground/background
    # split; a close-up with no real background (e.g. gill tissue filling
    # the whole frame) has no such split, so it arbitrarily divides the
    # image by internal shading into two or more comparably-sized pieces --
    # taking only the "largest" of those then crops out the rest of the fish.
    significant = [c for c in contours if cv2.contourArea(c) / total_area >= 0.05]
    if not significant:
        return 0, 0, w, h

    boxes = [cv2.boundingRect(c) for c in significant]
    x0 = min(bx for bx, by, bw, bh in boxes)
    y0 = min(by for bx, by, bw, bh in boxes)
    x1 = max(bx + bw for bx, by, bw, bh in boxes)
    y1 = max(by + bh for bx, by, bw, bh in boxes)

    area_ratio = ((x1 - x0) * (y1 - y0)) / total_area
    if area_ratio < 0.03:
        return 0, 0, w, h

    # cv2.findContours only ever traces the *foreground* (white) side of
    # Otsu's binary split -- the excluded half never becomes a contour to
    # union with in the first place. So also catch the split directly: a
    # region that touches both edges along one whole dimension (spans the
    # full width or height edge-to-edge) while covering a suspiciously
    # middling fraction of the frame is almost always an arbitrary
    # bisection of a no-real-background close-up, not a fish silhouette --
    # a genuine fish-on-background photo rarely has the fish running flush
    # to both opposite edges like that.
    spans_full_height = y0 <= 2 and y1 >= h - 2
    spans_full_width = x0 <= 2 and x1 >= w - 2
    if (spans_full_height or spans_full_width) and 0.25 <= area_ratio <= 0.85:
        return 0, 0, w, h

    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
    x = max(0, x0 - pad_x)
    y = max(0, y0 - pad_y)
    bw = min(w - x, bw + 2 * pad_x)
    bh = min(h - y, bh + 2 * pad_y)
    return x, y, bw, bh


# Standard ImageNet-1k class indices that are fish/fish-like aquatic species
# (verified against the canonical class list, not guessed). Used only as a
# coarse "does this even look like a fish" gate before running the disease
# classifier -- our 9-class model has no "not a fish" option of its own and
# will otherwise confidently force-fit any image (e.g. a human face) into
# one of its known classes.
FISH_IMAGENET_INDICES = {
    0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark",
    4: "hammerhead", 5: "electric ray", 6: "stingray", 389: "barracouta",
    390: "eel", 391: "coho salmon", 392: "rock beauty", 393: "anemone fish",
    394: "sturgeon", 395: "gar", 396: "lionfish", 397: "puffer",
}

_fish_gate_model = None


def _load_fish_gate_model():
    global _fish_gate_model
    if _fish_gate_model is None:
        _fish_gate_model = timm.create_model("mobilenetv3_small_100", pretrained=True)
        _fish_gate_model.eval()
    return _fish_gate_model


@torch.no_grad()
def fish_confidence(image_tensor: torch.Tensor) -> float:
    """Probability mass a generic ImageNet classifier puts on fish-like
    classes for this (already 224x224, ImageNet-normalized) image tensor.

    This is a coarse heuristic, not a validator: aquaculture species like
    tilapia/catfish aren't literally in ImageNet's 1000 classes, so a real
    disease photo may still only score moderately. It's tuned (via the
    caller's threshold) to catch clearly-not-a-fish input, not to reject
    unusual fish photos.
    """
    model = _load_fish_gate_model()
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    return float(sum(probs[i].item() for i in FISH_IMAGENET_INDICES))


class HybridFishDiseaseModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = CNN_BACKBONE,
        vit_backbone: str = VIT_BACKBONE,
        fusion_dim: int = FUSION_DIM,
        dropout: float = DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        # num_classes=0 -> no classifier head. global_pool='' on the CNN keeps
        # its output as a spatial (B, C, H, W) map for cnn_proj to pool itself.
        self.cnn = timm.create_model(cnn_backbone, pretrained=pretrained, num_classes=0, global_pool="")
        self.vit = timm.create_model(vit_backbone, pretrained=pretrained, num_classes=0)

        # Force the last ViT block off the fused-attention fast path so its
        # attention weights are actually computed and can be read back out
        # in disease_region() -- see _vit_attn_forward_with_capture().
        last_vit_attn = self.vit.blocks[-1].attn
        last_vit_attn.fused_attn = False
        last_vit_attn.forward = types.MethodType(_vit_attn_forward_with_capture, last_vit_attn)

        self.cnn_proj = nn.Sequential(
            nn.Linear(self.cnn.num_features, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(self.vit.num_features, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Cross-attention between the two modalities, then concatenate both
        # attended tokens ("Feature Concatenation + Attention Mechanism").
        self.attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_spatial = self.cnn(x)                                   # (B, C, h, w)
        cnn_pooled = torch.nn.functional.adaptive_avg_pool2d(cnn_spatial, 1).flatten(1)
        cnn_feat = self.cnn_proj(cnn_pooled)
        vit_feat = self.vit_proj(self.vit(x))

        tokens = torch.stack([cnn_feat, vit_feat], dim=1)          # (B, 2, fusion_dim)
        attended, _ = self.attn(tokens, tokens, tokens)             # (B, 2, fusion_dim)
        fused = attended.reshape(attended.size(0), -1)              # (B, 2 * fusion_dim)

        return self.classifier(fused)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def disease_region(self, x: torch.Tensor, crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
        """Where the ViT branch's last attention layer looked, as an (x, y, w, h)
        box in the crop's pixel space -- 14x14 resolution. This is the CLS
        token's attention over image patches: what the model's global summary
        drew on, not a class-conditional saliency map, and not a pixel-precise
        lesion segmentation -- there's no lesion-level training data to make
        a stronger claim, and it may not track the visible disease feature
        reliably on every photo.
        """
        self.forward(x)  # runs the patched last ViT block, populating its captured_attn

        attn = self.vit.blocks[-1].attn.captured_attn[0]  # (num_heads, N, N)
        cls_to_patches = attn[:, 0, 1:].mean(dim=0)         # (num_patches,), averaged over heads
        grid = int(round(cls_to_patches.numel() ** 0.5))
        cam = cls_to_patches.reshape(grid, grid)
        cam = cam / cam.max().clamp(min=1e-8)

        # 14x14 is finer than the CNN's 7x7 but still coarse next to a real
        # photo, so smooth before *and* after upscaling to avoid one noisy
        # cell dominating the box.
        cam_np = cam.cpu().numpy()
        cam_np = cv2.GaussianBlur(cam_np, (0, 0), sigmaX=0.7)
        cam_resized = cv2.resize(cam_np, (crop_w, crop_h))
        blur_sigma = max(crop_w, crop_h) * 0.03
        cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=blur_sigma)

        # Percentile threshold, not "% of max": attention weights are a
        # softmax over ~196 patches, so they're far more spread out than
        # Grad-CAM's activations, and last-layer ViT attention in particular
        # is well known to spread across several disjoint regions rather
        # than one blob. Taking the bounding box of *every* pixel above
        # threshold would then span almost the whole image -- instead, keep
        # only the single largest connected hot region.
        threshold = np.percentile(cam_resized, 80)
        mask = (cam_resized >= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, crop_w, crop_h
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)