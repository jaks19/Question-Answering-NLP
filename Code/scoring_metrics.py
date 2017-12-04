# Given a matrix of tuples [ [p+,p-,p--,...],[p++,p-,p--,...] ]
# Returns the MRR score for this set
def get_MRR_score(similarity_matrix):
    rows = similarity_matrix.split(1)
    reciprocal_ranks = []
    for r in rows:
        lst_scores = list(r[0].data)
        score_pos = lst_scores[0]
        lst_sorted_scores = sorted(lst_scores, reverse=True)
        rank = lst_sorted_scores.index(score_pos) + 1
        reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks)/len(reciprocal_ranks)