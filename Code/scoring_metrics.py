# Given a matrix of tuples [ [p+,p-,p--,...],[p++,p-,p--,...] ]
# Returns the MRR score for this set
def get_MRR_score(similarity_matrix):
    rows = similarity_matrix.split(1)
    reciprocal_ranks = []
    for r in rows:
        lst_scores = list(r[0])
        best_score = max(lst_scores)
        index_winner = lst_scores.index(best_score)
        if index_winner > 0: reciprocal_ranks.append(1.0/index_winner)
        else: reciprocal_ranks.append(0)
    return sum(reciprocal_ranks)/len(reciprocal_ranks)

