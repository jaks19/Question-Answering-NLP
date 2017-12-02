from preprocess import *
from get_q_matrices_functions import *

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def get_MRR_rank(dict_score_to_id, pos_qs_ids):
    scores = list(dict_score_to_id.keys())
    sorted_scores = sorted(scores, reverse=True)

    print(dict_score_to_id)
    print(pos_qs_ids)
    for i, score in enumerate(sorted_scores):
        if dict_score_to_id[score] in pos_qs_ids:
            return 1.0 / (i+1)
    return 0

def get_MRR_score(MRR_ranks):
    return(sum(MRR_ranks) / len(MRR_ranks))

''' Load trained model '''
lstm_loaded = torch.load('../Pickled/finalpooling_pairprincipleTrue_epoch4.pt')
lstm_loaded.eval()

''' Data Prep '''
# Organize eval data
dev = True
eval_data = devTest_id_to_similar_different(dev=dev)
eval_question_ids = list(eval_data.keys())
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData()

input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 240

''' Begin Testing '''

MRRs = []
total_qs = 0

for q in eval_question_ids:
    total_qs += 1

    # Build q's matrix
    q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(q, word2vec, id2Data)))

    # Get last hidden layer for q and normalize by num_words
    h0 = Variable(torch.zeros(1, 1, hidden_size))
    c0 = Variable(torch.zeros(1, 1, hidden_size))

    q_last_hidden = lstm_loaded(q_matrix_3d, (h0, c0))[1][0]
    normalized_q_hidden = q_last_hidden / q_matrix_3d.size()[1]

    # Grab associated questions
    all_qs = eval_data[q][1]

    # Keep list of positive question ids for use in MRR:
    pos_qs = eval_data[q][0]
    dict_score_to_id = {}

    for other_q in all_qs:

        # Init hidden layer and cell state
        h0 = Variable(torch.zeros(1, 1, hidden_size))
        c0 = Variable(torch.zeros(1, 1, hidden_size))

        # Build other_q's matrix
        other_q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(other_q, word2vec, id2Data)))

        # Get last hidden layer for other_q and normalize by num_words
        other_q_last_hidden = lstm_loaded(other_q_matrix_3d, (h0, c0))[1][0]
        normalized_other_q_hidden = other_q_last_hidden / other_q_matrix_3d.size()[1]

        # confirm
        dict_score_to_id[F.cosine_similarity(normalized_q_hidden.squeeze(0), normalized_other_q_hidden.squeeze(0), dim=1).data[0]] = other_q

    this_MRR = get_MRR_rank(dict_score_to_id, pos_qs)
    print(this_MRR)
    MRRs.append(this_MRR)

    print('MRR score right now: ', get_MRR_score(MRRs))