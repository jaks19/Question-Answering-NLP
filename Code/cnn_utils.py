import numpy as np
import torch
from torch.autograd import Variable


# Produces tensor [1 x num_words x input_size] for one particular question
def get_question_matrix(questionID, word2vec, id2Data, input_size, truncation_val):
    # Get the vector representation for each word in this question as list [v1,v2,v3,...]
    q_word_vecs = []
    for word in id2Data[questionID]:
        try:
            word_vec = np.array(word2vec[word]).astype(np.float32).reshape(len(word2vec[word]), -1)
            q_word_vecs.append(word_vec)
        except KeyError:
            pass

    # num_words x dim_words
    q_matrix = torch.Tensor(np.concatenate(q_word_vecs, axis=1).T)
    num_words_found = q_matrix.size()[0]
    if num_words_found < truncation_val:
        padding_rows = torch.zeros(truncation_val-num_words_found, input_size)
        q_matrix = torch.cat((q_matrix, padding_rows), 0)
    return [q_matrix.unsqueeze(0), num_words_found]


# Given ids of main qs in this batch
# Returns:
# 1. ids in ordered list as: 
# [ q_1+, q_1-, q_1--,..., q_1++, q_1-, q_1--,...,
# q_2+, q_2-, q_2--,..., q_2++, q_2-, q_2--,...,]
# All n main questions have their pos,neg,neg,neg,... interleaved
# 2. A dict mapping main question id --> its interleaved sequence length
def organize_ids_training(q_ids, data, num_differing_questions):
    sequence_ids = []
    dict_sequence_lengths = {}
    
    for q_main in q_ids:
        p_pluses = data[q_main][0]
        p_minuses = list(np.random.choice(data[q_main][1], num_differing_questions, replace = False))
        sequence_length = len(p_pluses) * num_differing_questions + len(p_pluses)
        dict_sequence_lengths[q_main] = sequence_length
        for p_plus in p_pluses:
            sequence_ids += [p_plus] + p_minuses

    return sequence_ids, dict_sequence_lengths


# Given ids of main qs in this batch
# Returns:
# 1. ids of the 20 questions for each q_main
# Note: Varying number of p_plus
# 2. A dict mapping main question id --> its p_pluses ids
def organize_test_ids(q_ids, data):
    sequence_ids = []
    dict_p_pluses = {}
    
    for i, q_main in enumerate(q_ids):
        all_p = data[q_main][1]
        p_pluses = data[q_main][0]
        p_pluses_indices = []
        for pos_id in p_pluses:
            p_pluses_indices += [all_p.index(pos_id)] 
        sequence_ids += all_p
        dict_p_pluses[i] = p_pluses_indices
        
    return sequence_ids, dict_p_pluses


# A tuple is (q+, q-, q--, q--- ...)
# Let all main questions be set Q
# Each q in Q has a number of tuples equal to number of positives |q+, q++, ...|
# Each q in Q will have a 2D matrix of: num_tuples x num_candidates_in_tuple
# Concatenate this matrix for all q in Q and you get a matrix of: |Q| x num_tuples x num_candidates_in_tuple

# The above is for candidates
# To do cosine_similarity, need same structure with q's
# Basically each q will be a matrix of repeated q's: num_tuples x num_candidates_in_tuple, all elts are q (repeated)

# This method constructs those matrices, use candidates=True for candidates matrix
def construct_qs_matrix_training(q_ids_sequential, cnn, word2vec, id2Data, dict_sequence_lengths, input_size, num_differing_questions, truncation_val, candidates=False):
    if not candidates:
        q_ids_complete = []
        for q in q_ids_sequential:
            q_ids_complete += [q] * dict_sequence_lengths[q]
    
    else: q_ids_complete = q_ids_sequential

    qs_matrix_list = []
    qs_seq_length = []
    
    for q in q_ids_complete:
        q_matrix_3d, q_num_words = get_question_matrix(q, word2vec, id2Data, input_size, truncation_val)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_num_words)

    qs_padded = Variable(torch.cat(qs_matrix_list, 0))
    qs_hidden = cnn(torch.transpose(qs_padded, 1, 2))
    sum_h_qs = torch.sum(qs_hidden, dim=2)
    mean_pooled_h_qs = torch.div(sum_h_qs, torch.autograd.Variable(torch.FloatTensor(qs_seq_length)[:, np.newaxis]))
    qs_tuples = mean_pooled_h_qs.split(1+num_differing_questions)
    final_matrix_tuples_by_constituent_qs_by_hidden_size = torch.stack(qs_tuples, dim=0, out=None)
    
    return final_matrix_tuples_by_constituent_qs_by_hidden_size


# Case candidates: gives a matrix with a row for each q_main, with 20 p's
# Case not candidates: gives a matrix with a row for each q_main, with 20 q_main's repeated
def construct_qs_matrix_testing(q_ids_sequential, cnn, word2vec, id2Data, input_size, num_differing_questions, truncation_val, candidates=False):
    if not candidates:
        q_ids_complete = []
        for q in q_ids_sequential:
            q_ids_complete += [q] * num_differing_questions
    
    else: q_ids_complete = q_ids_sequential

    qs_matrix_list = []
    qs_seq_length = []
    
    for q in q_ids_complete:
        q_matrix_3d, q_num_words = get_question_matrix(q, word2vec, id2Data, input_size, truncation_val)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_num_words)

    qs_padded = Variable(torch.cat(qs_matrix_list, 0))
    qs_hidden = cnn(torch.transpose(qs_padded, 1, 2))
    sum_h_qs = torch.sum(qs_hidden, dim=2)
    mean_pooled_h_qs = torch.div(sum_h_qs, torch.autograd.Variable(torch.FloatTensor(qs_seq_length)[:, np.newaxis]))
    qs_tuples = mean_pooled_h_qs.split(num_differing_questions)
    final_matrix_tuples_by_constituent_qs_by_hidden_size = torch.stack(qs_tuples, dim=0, out=None)
    
    return final_matrix_tuples_by_constituent_qs_by_hidden_size