import numpy as np
import torch
from torch import nn
import logging

import pickle
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import utils.config as cnf

logger = logging.getLogger(__name__)

def find_peak(x, threshold=None):
    # x: (ch, y_size, x_size) numpy array
    ch, y_size, x_size = x.shape
    tmp = np.zeros((9, ch, y_size + 2, x_size + 2))
    tmp[0, :, 1:-1, 1:-1] = x
    tmp[1, :, 0:-2, 0:-2] = x
    tmp[2, :, 0:-2, 1:-1] = x
    tmp[3, :, 0:-2, 2:] = x
    tmp[4, :, 1:-1, 0:-2] = x
    tmp[5, :, 1:-1, 2:] = x
    tmp[6, :, 2:, 0:-2] = x
    tmp[7, :, 2:, 1:-1] = x
    tmp[8, :, 2:, 2:] = x
    peak_map = (tmp[0] > np.max(tmp[1:], axis=0))[:, 1:-1, 1:-1]
    if threshold:
        peak_map = np.logical_and(peak_map, (x > threshold))
    peak_index_list = []
    for p in peak_map:
        peak_index_list.append(np.transpose(p.nonzero()))
    return peak_index_list # (2, 164, 166)


@torch.no_grad()
def rematch_vector(center_pred, movement_pred, peak_threshold, match_threshold_pixel=70):
    quarter = 1000
    euclid_threshold=int(match_threshold_pixel)

    first_img_peak, last_img_peak = find_peak(center_pred[:2], threshold = peak_threshold)
    # displacement = (np.transpose(movement_pred[:, first_img_peak[:, 0], first_img_peak[:, 1]])).astype(int)
    displacement = movement_pred.astype(int)

    first_peak_dict = {i: list(p) for i, p in enumerate(first_img_peak)}
    last_peak_dict = {i+quarter: list(p) for i, p in enumerate(last_img_peak)}

    if len(first_peak_dict) == 0 or len(last_peak_dict) == 0:
        matched_first_peak=np.empty([0, 2], dtype = np.uint8)
        matched_last_peak=np.empty([0, 2], dtype = np.uint8)
        change_displacement = np.empty([0, 2], dtype = np.uint8)
        return matched_first_peak, matched_last_peak, change_displacement

    G = nx.Graph()
    G.add_nodes_from(first_peak_dict.keys(), bipartite=1)
    G.add_nodes_from(last_peak_dict.keys(), bipartite=0)

    weight, euclid, cosine = calculate_weight(first_img_peak, last_img_peak, displacement, euclid_threshold)
    weight_df = pd.DataFrame(weight,
                       index = list(first_peak_dict.keys()), columns = list(last_peak_dict.keys()))

    mask = np.argwhere(weight_df.to_numpy() <= 5.0) # find true index
    mask[:, 1] += quarter
    mask = [tuple(m) for m in mask]

    weight_list = []
    for last in last_peak_dict.keys():
        for idx, row in weight_df.iterrows():
            weight_list.append((idx, last, row[last]))

    G.add_weighted_edges_from(weight_list)
    matched = bipartite.matching.minimum_weight_full_matching(G, weight='weight')
    matched = {i: matched[i] for i in matched.keys() if i < quarter}
    matched = {item[0]: item[1] for item in matched.items() if item in mask}

    matched_first_peak = np.array([first_peak_dict[k] for k in matched.keys()])
    matched_last_peak = np.array([last_peak_dict[v] for v in matched.values()])
    if len(matched_first_peak) == 0: matched_first_peak = np.empty([0,2], dtype = np.uint8)
    if len(matched_last_peak) == 0: matched_last_peak = np.empty([0,2], dtype = np.uint8)

    change_displacement = matched_last_peak - matched_first_peak

    return matched_first_peak, matched_last_peak, change_displacement

def calculate_weight(first_img_peak, last_img_peak, pred_displacement, euclid_threshold):
    end_point = first_img_peak + pred_displacement
    euclid_distance = calculate_euclid_distance(end_point, last_img_peak, euclid_threshold)
    cosine_distance, length_distance = calculate_cosine_distance(pred_displacement, first_img_peak, last_img_peak)
    alpha = 1.0 # euclid_distance_weight
    beta = 1.0 # cosine_distance_weight
    gamma = 1.0 # length_distance_weight
    distance = alpha * euclid_distance + beta * cosine_distance + gamma * length_distance
    return distance, euclid_distance, cosine_distance

def calculate_euclid_distance(end_point, last_img_peak, euclid_threshold):
    euclid_distance = np.power((end_point[:,0, np.newaxis] - last_img_peak[:,0]), 2) + np.power((end_point[:,1, np.newaxis] - last_img_peak[:,1]), 2)
    euclid_distance = euclid_distance/euclid_threshold
    euclid_distance = np.where(euclid_distance >= 5, 10, euclid_distance)
    return euclid_distance

def calculate_cosine_distance(pred_displacement, first_img_peak, last_img_peak):
    c0 = last_img_peak[:, np.newaxis, 0] - first_img_peak[:, 0] # subtract by x axis
    c1 = last_img_peak[:, np.newaxis, 1] - first_img_peak[:, 1] # subtract by y axis
    last2first_displacement=np.dstack((c1, c0))  # (l, f, 2)
    ln, fn, _ = last2first_displacement.shape

    cosine_similarity = np.zeros((ln, fn), dtype = np.float32)
    length_ratio = np.zeros((ln, fn), dtype = np.float32)
    pred_displacement_norm = np.linalg.norm(pred_displacement, axis=1)

    # Avoid that divide by zero
    last2first_displacement = np.where(last2first_displacement==0, 1e-16, last2first_displacement)
    pred_displacement_norm = np.where(pred_displacement_norm==0, 1e-16, pred_displacement_norm)

    for c in range(ln):
        numerator = np.dot(pred_displacement, last2first_displacement[c].T).diagonal()
        denominator = np.linalg.norm(last2first_displacement[c], axis=1) * pred_displacement_norm
        length_r = np.linalg.norm(last2first_displacement[c], axis=1) / pred_displacement_norm
        cosine_col = numerator / denominator
        cosine_similarity[c] = cosine_col
        length_ratio[c] = length_r

    cosine_distance = 1 - cosine_similarity.T
    cosine_distance = np.where(cosine_distance >= 0.2, 5, 0) # penalty for 30 ~ 180

    length_distance = np.abs(length_ratio - 1).T
    length_distance = np.where(length_distance >= 0.7, 5, 0)

    return cosine_distance, length_distance

def nms_keypoint(nms_threshold=0.3):
    pass

if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    with open('C:\dev\Computer_Vision\pred.pkl', 'rb') as f:
        pred_pkl = pickle.load(f)

    print(f'pred_pkl.shape: {pred_pkl.shape}') # ([2, 4, 160, 160])

    class_num = len(cnf.merge_category_dict)

