# utils/loss.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ramp_weight(epoch: int, warmup: int, ramp: int, max_w: float) -> float:
    """
    epoch: 0-based
    warmup: số epoch đầu weight=0
    ramp: số epoch tăng dần từ 0 -> max_w
    """
    if max_w == 0:
        return 0.0
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return float(max_w)
    t = (epoch - warmup) / float(ramp)  # 0..1
    t = max(0.0, min(1.0, t))
    return float(max_w) * t


class CLIPCAERLoss(nn.Module):
    """
    L = CE + w_mi(epoch)*MI + w_dc(epoch)*DC
    - CE: optional label smoothing
    - MI: InfoNCE-ish using your mi_estimator(pos-neg)
    - DC: KL( P_joint || P_l ⊗ P_h ) theo paper, tính từ logits 2 view
    """

    def __init__(self, args, mi_estimator=None, num_classes=5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mi_estimator = mi_estimator

        # base lambdas (max weight)
        self.lambda_mi = float(getattr(args, "lambda_mi", 1.0))
        self.lambda_dc = float(getattr(args, "lambda_dc", 0.0))

        # warmup/ramp
        self.mi_warmup = int(getattr(args, "mi_warmup", 0))
        self.mi_ramp   = int(getattr(args, "mi_ramp", 0))
        self.dc_warmup = int(getattr(args, "dc_warmup", 0))
        self.dc_ramp   = int(getattr(args, "dc_ramp", 0))

        # label smoothing
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.0))

        # optional class weights (nếu bạn muốn dùng)
        cw = getattr(args, "class_weights", None)  # list/tuple or None
        if cw is not None:
            cw = torch.tensor(cw, dtype=torch.float32)
        self.register_buffer("class_weights", cw if cw is not None else None)

        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # cache weights for printing (trainer có thể đọc)
        self.last_w_mi = 0.0
        self.last_w_dc = 0.0

    def set_epoch(self, epoch: int):
        """Gọi mỗi epoch để cập nhật weight MI/DC."""
        w_mi = _ramp_weight(epoch, self.mi_warmup, self.mi_ramp, self.lambda_mi)
        w_dc = _ramp_weight(epoch, self.dc_warmup, self.dc_ramp, self.lambda_dc)
        self.last_w_mi = w_mi
        self.last_w_dc = w_dc

    # ---------------- utils ----------------
    def _sanitize_targets(self, targets):
        if targets.dim() > 1:
            targets = targets.view(-1)
        return targets.long().clamp(0, self.num_classes - 1)

    # ---------------- MI ----------------
    def _mi_loss(self, f_l, f_h):
        """
        f_l: learnable_text_features [C,D] hoặc [B,D]
        f_h: handcrafted_text_features [C,D] hoặc [B,D]
        Với code hiện tại của bạn: text_features thường [C,D]
        -> MI ở mức "set-wise" vẫn chạy, nhưng ổn định hơn nếu float32.
        """
        if (
            f_l is None
            or f_h is None
            or self.mi_estimator is None
            or self.last_w_mi == 0.0
        ):
            # trả về 0 đúng dtype/device
            if isinstance(f_l, torch.Tensor):
                return f_l.new_tensor(0.0)
            if isinstance(f_h, torch.Tensor):
                return f_h.new_tensor(0.0)
            return torch.tensor(0.0)

        f_l = f_l.float()
        f_h = f_h.float()

        pos = self.mi_estimator(f_l, f_h).mean()
        idx = torch.randperm(f_h.size(0), device=f_h.device)
        neg = self.mi_estimator(f_l, f_h[idx]).mean()

        return -(pos - neg)

    # ---------------- DC ----------------
    def _dc_loss(self, logits_l, logits_h, eps=1e-8):
        """
        DC theo paper (ổn định):
          p_l = softmax(logits_l)
          p_h = softmax(logits_h)
          P = sum_b p_l(b) ⊗ p_h(b)  (joint)
          dc = KL( P || P_l ⊗ P_h )
        """
        if logits_l is None or logits_h is None or self.last_w_dc == 0.0:
            if isinstance(logits_l, torch.Tensor):
                return logits_l.new_tensor(0.0)
            if isinstance(logits_h, torch.Tensor):
                return logits_h.new_tensor(0.0)
            return torch.tensor(0.0)

        p_l = F.softmax(logits_l.float(), dim=1)
        p_h = F.softmax(logits_h.float(), dim=1)

        P = torch.einsum("bi,bj->ij", p_l, p_h)
        P = P / (P.sum() + eps)

        P_l = P.sum(dim=1, keepdim=True)
        P_h = P.sum(dim=0, keepdim=True)

        P   = P.clamp_min(eps)
        P_l = P_l.clamp_min(eps)
        P_h = P_h.clamp_min(eps)

        dc = (P * (torch.log(P) - torch.log(P_l) - torch.log(P_h))).sum()
        return dc

    # ---------------- Forward ----------------
    def forward(
        self,
        logits,                       # learnable logits [B,C]
        targets,
        *,
        epoch: int = None,            # <-- pass epoch để tự set weight
        learnable_text_features=None,
        hand_crafted_text_features=None,
        logits_hand=None,             # handcrafted logits [B,C]
    ):
        if epoch is not None:
            self.set_epoch(int(epoch))

        targets = self._sanitize_targets(targets)
        ce = self.ce_loss(logits, targets)

        mi = self._mi_loss(learnable_text_features, hand_crafted_text_features)
        dc = self._dc_loss(logits, logits_hand)

        total = ce + self.last_w_mi * mi + self.last_w_dc * dc

        return {
            "total": total,
            "ce": ce,
            "mi": mi,
            "dc": dc,
            "w_mi": float(self.last_w_mi),
            "w_dc": float(self.last_w_dc),
        }


def build_criterion(args, mi_estimator=None, num_classes=5):
    return CLIPCAERLoss(args, mi_estimator=mi_estimator, num_classes=num_classes)