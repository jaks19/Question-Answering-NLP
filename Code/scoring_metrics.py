import numpy as np
import sklearn.metrics


# Given a matrix of similarities, one row for each q_main
# Returns the MRR score for this set
# Note: each q_main may have anywhere from 0 to 20 positives, not including 20
def get_MRR_score(similarity_matrix, dict_pos):
    rows = similarity_matrix.split(1)
    reciprocal_ranks = []

    for row_index, r in enumerate(rows):
        pos_indices_this_row = dict_pos[row_index]
        lst_scores = list(r[0].data)
        rank = None
        lst_sorted_scores = sorted(lst_scores, reverse=True)
        
        for rk, score in enumerate(lst_sorted_scores):
            index_original = lst_scores.index(score)
            if index_original in pos_indices_this_row:
                rank = rk + 1
                reciprocal_ranks.append(1.0 / rank)
                break
        
    return sum(reciprocal_ranks)/len(reciprocal_ranks)


def get_MAP_score(similarity_matrix, dict_pos):
    rows = similarity_matrix.split(1)
    avg_precision_scores = []

    for row_index, r in enumerate(rows):
        pos_indices_this_row = dict_pos[row_index]
        lst_scores = list(r[0].data)
        lst_sorted_scores = sorted(lst_scores, reverse=True)
        
        qs_binary = []
        for score in lst_sorted_scores:
            index_original = lst_scores.index(score)
            if index_original in pos_indices_this_row:
                qs_binary.append(1)
            else:
                qs_binary.append(0)
                
        if sum(qs_binary) == 0:
            pass
        else:
            avg_precision_scores.append(sklearn.metrics.average_precision_score(np.array(qs_binary), np.array(lst_sorted_scores)))
        
    return sum(avg_precision_scores)/len(avg_precision_scores)

# gets avg precision metric for q
#def get_MAP_scores(dict_score_to_id, pos_qs_ids):
    #qs_binary = []
    #scores = []
    #for score in dict_score_to_id.keys():
        #scores.append(score)
#
        #if dict_score_to_id[score] in pos_qs:
            #qs_binary.append(1)
        #else:
            #qs_binary.append(0)
    
    #if sum(qs_binary) == 0:
        #return None
    
    #return sklearn.metrics.average_precision_score(np.array(qs_binary), np.array(scores))

# mean of average precision metric across qs
#def get_MAP_score(APs):
    #return sum(APs)/len(APs)


    
def avg_precision_at_k(similarity_matrix, dict_pos, k):
    rows = similarity_matrix.split(1)
    precisions_at_k = []

    for row_index, r in enumerate(rows):
        pos_indices_this_row = dict_pos[row_index]
        lst_scores = list(r[0].data)
        lst_sorted_scores = sorted(lst_scores, reverse=True)
        
        top_qs_binary = []
        for i in range(k):
            index_original = lst_scores.index(lst_sorted_scores[i])
            if index_original in pos_indices_this_row:
                top_qs_binary.append(1)
            else:
                top_qs_binary.append(0)
                
        precisions_at_k.append(sum(top_qs_binary)/len(top_qs_binary))
        
    return sum(precisions_at_k)/len(precisions_at_k)

# precision for a q through the kth ranked +-question
#def precision_at_k(dict_score_to_id, pos_qs_ids, k):
#    scores = list(dict_score_to_id.keys())
#    sorted_scores = sorted(scores, reverse=True)
    
#    top_qs_binary = []
#    for i in range(k):
#        if dict_score_to_id[sorted_scores[i]] in pos_qs_ids:
#            top_qs_binary.append(1)
#        else:
#            top_qs_binary.append(0)
            
#    return sum(top_qs_binary)/len(top_qs_binary)

# average precisions at k across qs
#def avg_precision_at_k(precisions_at_k):
#    return sum(precisions_at_k)/len(precisions_at_k)

