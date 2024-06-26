import torch.nn as nn
import torch
import torch.nn.functional as F

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)  # (64,1)
    t2 = (t * mask2).sum(1, keepdims=True)  # (64,1)
    rt = torch.cat([t1, t2], dim=1)  # (64,2)
    return rt

#"Discoupled Knowledge Distillation" "CVPR2022"
class DKD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, logits_student, logits_teacher, target, alpha, beta, temperature):
        ### 获得每个target值对应的掩码，从而获得p_t
        gt_mask = _get_gt_mask(logits_student, target)
        ### 获得其他target对应的掩码，从而获得p_{\t}
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        ## 计算b^T以及b^S
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss
