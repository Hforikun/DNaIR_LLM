import itertools

import numpy as np


def ils_metric(rec_list, item_sim_matrix):
    sim_temp = 0
    for i in range(0, len(rec_list)):
        for j in range(i + 1, len(rec_list)):
            if str(rec_list[j]) in item_sim_matrix[str(rec_list[i])]:
                sim_temp += item_sim_matrix[str(rec_list[i])][str(rec_list[j])]
    return 1 - (sim_temp / (len(rec_list) * (len(rec_list) - 1)))


def ndcg_metric(rec_list, test_dict):
    ndcg = 0
    for key, topn_set in rec_list.items():
        test_set = test_dict.get(key)
        dsct_list = [1 / np.log2(i + 1) for i in range(1, len(topn_set) + 1)]
        z_k = sum(dsct_list)
        if test_set is not None:
            mask = [0 if i not in test_set else 1 for i in topn_set]
            ndcg += sum(np.multiply(dsct_list, mask)) / z_k
    ndcg = ndcg / len(rec_list.items())
    return ndcg


def novelty_metric(rec_list, pop_dict):
    pop_sum = []
    for item in rec_list:
        if str(item) in pop_dict.keys():
            pop_sum.append(pop_dict[str(item)])
    return np.mean(pop_sum)


def interdiv_metric(interdiv_list):
    interdiv_result = 0
    interdiv_comb_list = list(itertools.combinations(interdiv_list, 2))
    for each_comb in interdiv_comb_list:
        temp_comb = []
        temp_comb.extend(each_comb[0])
        temp_comb.extend(each_comb[1])
        interdiv_result += len(list(set(each_comb[0]) & set(each_comb[1]))) / (len(each_comb[0]))
    return 2 * interdiv_result / len(interdiv_comb_list) if interdiv_comb_list else 0

def recall_metric(rec_list_dict, test_dict):
    """Recall@K"""
    recall = 0
    for key, topn_set in rec_list_dict.items():
        test_set = test_dict.get(key)
        if test_set and len(test_set) > 0:
            hit = len(set(topn_set) & set(test_set))
            recall += hit / len(set(test_set))
    return recall / len(rec_list_dict) if rec_list_dict else 0

def ltc_metric(all_recommended_items, tail_items_set):
    """Long-tail Coverage (LtC)
    The fraction of unique long-tail items exposed to users relative to total long-tail items.
    """
    if not tail_items_set:
        return 0
    exposed_tail = set(all_recommended_items) & set(tail_items_set)
    return len(exposed_tail) / len(tail_items_set)

def mrmc_metric(rec_lists, head_items_set, target_p=None):
    """Mean Rank Miscalibration (MRMC).
    Measures the divergence between the actual distribution of head/tail items in recommendations
    and a target distribution (e.g., matching the background catalog 20% head / 80% tail).
    Lower is better (fairer).
    """
    if target_p is None:
        target_p = [0.2, 0.8]  # [head_ratio, tail_ratio] target
    
    mrmc_sum = 0
    for rec in rec_lists:
        if not rec:
            continue
        head_count = sum(1 for item in rec if item in head_items_set)
        tail_count = len(rec) - head_count
        
        q_head = max(head_count / len(rec), 1e-6)
        q_tail = max(tail_count / len(rec), 1e-6)
        
        # KL divergence KL(target || q)
        kl = target_p[0] * np.log(target_p[0] / q_head) + target_p[1] * np.log(target_p[1] / q_tail)
        
        # Worst-case divergence (user gets 99.99% head items despite 20% target)
        kl_worst = target_p[0] * np.log(target_p[0] / 0.9999) + target_p[1] * np.log(target_p[1] / 0.0001)
        
        mc = kl / kl_worst
        mrmc_sum += mc
        
    return mrmc_sum / len(rec_lists) if rec_lists else 0
