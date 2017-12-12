import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics

def get_MRR_rank(dict_score_to_id, pos_qs_ids):
    scores = list(dict_score_to_id.keys())
    sorted_scores = sorted(scores, reverse=True)

    for i, score in enumerate(sorted_scores):
        if dict_score_to_id[score] in pos_qs_ids:
            return 1.0 / (i+1)
    return 0

def get_MRR_score(MRR_ranks):
    return(sum(MRR_ranks) / len(MRR_ranks))


# gets avg precision metric for q
def get_AP(dict_score_to_id, pos_qs_ids):
	qs_binary = []
	scores = []
	for score in dict_score_to_id.keys():
		scores.append(score)
	
		if dict_score_to_id[score] in pos_qs:
			qs_binary.append(1)
		else:
			qs_binary.append(0)
    
	if sum(qs_binary) == 0:
		return None
    
	return sklearn.metrics.average_precision_score(np.array(qs_binary), np.array(scores))

# mean of average precision metric across qs
def get_MAP_score(APs):
	return sum(APs)/len(APs)


# precision for a q through the kth ranked +-question
def precision_at_k(dict_score_to_id, pos_qs_ids, k):
	scores = list(dict_score_to_id.keys())
	sorted_scores = sorted(scores, reverse=True)
   
	top_qs_binary = []
	for i in range(k):
		if dict_score_to_id[sorted_scores[i]] in pos_qs_ids:
			top_qs_binary.append(1)
		else:
			top_qs_binary.append(0)
            
	return sum(top_qs_binary)/len(top_qs_binary)

# average precisions at k across qs
def avg_precision_at_k(precisions_at_k):
	return sum(precisions_at_k)/len(precisions_at_k)