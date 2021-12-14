import numpy as np
import torch
from torch import nn
from utils.helper import rematch_vector, find_peak
from utils.config import kpt_oks_sigmas
import utils.config as cnf


class Metric:
    def __init__(self, device, img_resolution, num_batch, class_num=len(cnf.merge_category_dict), peak_threshold=0.2, oks_threshold=0.5):
        self.variances = (kpt_oks_sigmas * 2) ** 2
        self.class_num = class_num
        self.img_resolution = img_resolution
        self.num_batch = num_batch
        self.peak_threshold = peak_threshold
        self.oks_threshold = oks_threshold
        self.device = device

    @torch.no_grad()
    def get_batch_statistics(self, pred, target):
        target_dist, target_displacement = target
        target_dist = torch.tensor(target_dist, device = self.device, dtype = torch.float32)
        num_batch, _, _, _ = pred.shape

        cosine_similarity_batch = np.zeros((self.class_num,), dtype = np.float32)

        first_peak_metrics = np.zeros((self.class_num, 5,), dtype = np.float32) # oks_avg, precision, recall, ap, f1-score
        last_peak_metrics=np.zeros((self.class_num, 5,), dtype = np.float32)

        for b in range(0, num_batch):
            center_pred = pred[b, :self.class_num*2, :, :]
            movement_pred = pred[b, self.class_num*2:, :, :].cpu().numpy()
            center_target = target_dist[b, :self.class_num*2, :, :]
            movement_target = target_displacement[b]

            for c in range(0, self.class_num):
                center_pred_class = nn.Sigmoid()(center_pred[c*2:c*2+2, :, :]).cpu().numpy()
                center_target_class = nn.Sigmoid()(center_target[c*2:c*2+2, :, :]).cpu().numpy()

                movement_target_class=movement_target[:, c*4 + 2:c*4 + 4]
                movement_pred_class=movement_pred[c*2:c*2+2, :, :]

                pred_first_peak, pred_last_peak = find_peak(center_pred_class, threshold = self.peak_threshold)
                target_first_peak, target_last_peak = find_peak(center_target_class, threshold = self.peak_threshold)
                movement_pred_class = (np.transpose(movement_pred_class[:, pred_first_peak[:, 0], pred_first_peak[:, 1]]))

                pred_first_peak, pred_last_peak, movement_pred_class =\
                    rematch_vector(center_pred_class, movement_pred_class, peak_threshold = self.peak_threshold)

                pred_first_objectness=center_pred_class[0, pred_first_peak[:, 0], pred_first_peak[:, 1]]
                pred_last_objectness=center_pred_class[0, pred_last_peak[:, 0], pred_last_peak[:, 1]]
                cosine_similarity = compute_displacement_similarity(movement_pred_class, movement_target_class, pred_first_peak)

                first_peak_m = self.get_statistics_per_peak(pred_first_peak, target_first_peak, pred_first_objectness, variance = self.variances[c])
                last_peak_m = self.get_statistics_per_peak(pred_last_peak, target_last_peak, pred_last_objectness, variance = self.variances[c])

                first_peak_metrics[c] += first_peak_m
                last_peak_metrics[c] += last_peak_m
                cosine_similarity_batch[c] += cosine_similarity

        first_peak_metrics /= num_batch
        last_peak_metrics /= num_batch
        cosine_similarity_batch /= num_batch

        return first_peak_metrics, last_peak_metrics, cosine_similarity_batch

    def get_statistics_per_peak(self, pred_peak, target_peak, objectness, variance):
        pred_num, _ = pred_peak.shape
        target_num, _ = target_peak.shape

        if pred_num == 0 or target_num == 0:
            return np.array([0, 0, 0, 0, 0])

        peak_oks_avg, peak_oks_list = compute_oks(pred_peak, target_peak, pred_num, variance)

        tps = np.zeros((pred_num,), dtype = np.int8)
        tps[peak_oks_list>=self.oks_threshold] = 1
        p, r, ap, f1 = ap_per_class(tps, objectness, target_num)

        return np.array([peak_oks_avg, p, r, ap, f1])


def compute_oks(pred_peak, target_peak, count, variance, img_size = (160, 160)):
    count_target, _ = target_peak.shape

    axis = 1 if count == count_target else 0

    x_tg = target_peak[:, 0]
    y_tg = target_peak[:, 1]
    x_pred = pred_peak[:, 0]
    y_pred = pred_peak[:, 1]

    area = img_size[0] * img_size[1]
    squared_distance = (x_tg[:, np.newaxis] - x_pred) ** 2 + (y_tg[:, np.newaxis] - y_pred) ** 2 # (target_num, pred_num)
    squared_distance = squared_distance / (area * variance * 2)
    squared_distance = np.min(squared_distance, axis=axis)

    oks = np.exp(-squared_distance, dtype = np.float32)
    oks_avg = np.sum(oks) / count
    return oks_avg, oks

def compute_displacement_similarity(pred_displacement, target_displacement, pred_first_img_peak):
    target_displacement=target_displacement.astype(int)
    pred_displacement = pred_displacement.astype(int)

    tn, _=target_displacement.shape
    pn, _=pred_displacement.shape
    if pn == 0:
        return 0
    count=max(tn, pn)
    axis=1 if count == tn else 0

    target_displacement_norm = np.linalg.norm(target_displacement, axis=1)
    pred_displacement_norm = np.linalg.norm(pred_displacement, axis=1)

    # Avoid that divide by zero
    target_displacement_norm = np.where(target_displacement_norm==0, 1e-05, target_displacement_norm)
    pred_displacement_norm = np.where(pred_displacement_norm==0, 1e-05, pred_displacement_norm)

    numerator = np.dot(target_displacement, pred_displacement.T)
    denominator = target_displacement_norm[:, np.newaxis] * pred_displacement_norm
    cosine_similarity = numerator / denominator

    cosine_similarity = np.where(cosine_similarity <= 0, 0, cosine_similarity)
    cosine_similarity = np.max(cosine_similarity, axis=axis)
    cosine_similarity = np.sum(cosine_similarity)
    cosine_similarity /= count

    return cosine_similarity


def ap_per_class(tps, objectness, targets_num):
    """
    :param tps: (list)
    :param objectness: (list)
    :param targets_num
    :return:
    """
    i = np.argsort(-objectness)
    tps, objectness = tps[i], objectness[i]

    fpc = (1 - tps).cumsum()
    tpc = tps.cumsum()
    recall_curve = tpc / (targets_num + 1e-16)
    recall = recall_curve[-1]

    precision_curve = tpc / (tpc + fpc)
    precision= precision_curve[-1]

    ap = compute_ap(recall_curve, precision_curve)

    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    return precision, recall, ap, f1

def compute_ap(recall, precision):
    """
    :param recall: (list)
    :param precision: (list)
    :return: ap
    """
    mrec = np.concatenate(([0,0], recall, [1,0]))
    mpre = np.concatenate(([0,0], precision, [0,0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap

def count_axis(pred_peak, target_peak):
    count_target, _ = target_peak.shape
    count_pred, _ = pred_peak.shape
    count_max = max(count_pred, count_target)
    axis = 1 if count_max == count_target else 0

    return count_pred, count_target, count_max, axis