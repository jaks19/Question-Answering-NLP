from preprocess import *
from get_q_matrices_functions import *

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity


''' Data Prep '''
training_data = training_id_to_similar_different()
trainingQuestionIds = list(training_data.keys())
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData_truncate(100)


''' Model Specs '''
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
dropout = 0.2
bidirectional = False

lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function = torch.nn.MultiMarginLoss(margin=0.2)
optimizer = torch.optim.Adam(lstm.parameters(), lr=10**-4, weight_decay=0.001)

h0 = Variable(torch.zeros(1, 1, hidden_size), requires_grad=True)
c0 = Variable(torch.zeros(1, 1, hidden_size), requires_grad=True)


''' Procedural parameters '''
batch_size = 100
num_differing_questions = 20

num_epochs = 10
num_batches = round(len(training_data.keys())/batch_size)


'''Matrix constructors (use global vars, leave in order)'''

# Given a list of ids, compute the hidden layer for each of those questions
# Ideal if need to work on a group, not just one question
def mean_pooled_hidden_layers_for_ids(list_ids, input_size):
    qs_matrix_list = []
    qs_seq_length = []
        
    for q in list_ids:
        q_matrix_3d = get_question_matrix(q, word2vec, id2Data)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_matrix_3d.shape[1])
    
    qs_padded = padded_q_matrix(qs_seq_length, qs_matrix_list, input_size)
    qs_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(qs_padded, (h0, c0))[0], batch_first=True)
    sum_h_qs = torch.sum(qs_hidden[0], dim=1)
    
    lst_hidden = []
    for i in range(len(sum_h_qs)): 
        h = sum_h_qs[i] / qs_seq_length[i]
        lst_hidden.append(h)
    return lst_hidden

# Given ids of main qs in this batch
#
# Returns:
# 1. ids in ordered list as: 
# [
# q_1+, q_1-, q_1--,..., q_1++, q_1-, q_1--,...,
# q_2+, q_2-, q_2--,..., q_2++, q_2-, q_2--,...,
# ...
# ]
# All n main questions have their pos,neg,neg,neg,... interleaved
#
# 2. A dict mapping main question id --> its interleaved sequence length

def order_ids(q_ids):
    global training_data
    global num_differing_questions
    
    sequence_ids = []
    dict_sequence_lengths = {}
    
    for q_main in q_ids:
        p_pluses = training_data[q_main][0]
        p_minuses = list(np.random.choice(training_data[q_main][1], num_differing_questions, replace = False))
        sequence_length = len(p_pluses) * num_differing_questions + len(p_pluses)
        dict_sequence_lengths[q_main] = sequence_length
        for p_plus in p_pluses:
            sequence_ids += [p_plus] + p_minuses

    return sequence_ids, dict_sequence_lengths

# sequence_ids, dict_sequence_lengths = order_ids([193,295137])



# A tuple is (q+, q-, q--, q--- ...)
# Let all main questions be set Q
# Each q in Q has a number of tuples equal to number of positives |q+, q++, ...|
# Each q in Q will have a 2D matrix of: num_tuples x num_candidates_in_tuple
# Concatenate this matrix for all q in Q and you get a matrix of: |Q| x num_tuples x num_candidates_in_tuple

# The above is for candidates
# To do cosine_similarity, need same structure with q's
# Basically each q will be a matrix of repeated q's: num_tuples x num_candidates_in_tuple, all elts are q (repeated)

# This method constructs those matrices, use candidates=True for candidates matrix

def construct_qs_matrix(q_ids_sequential, dict_sequence_lengths, candidates=False):
    global lstm, h0, c0, word2vec, id2data, input_size, num_differing_questions
    
    if not candidates:
        q_ids_complete = []
        for q in q_ids_sequential:
            q_ids_complete += [q] * dict_sequence_lengths[q]
    
    else: q_ids_complete = q_ids_sequential

    qs_matrix_list = []
    qs_seq_length = []
    
    for q in q_ids_complete:
        q_matrix_3d = get_question_matrix(q, word2vec, id2Data)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_matrix_3d.shape[1])

    qs_padded = padded_q_matrix(qs_seq_length, qs_matrix_list, input_size)
    qs_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(qs_padded, (h0, c0))[0], batch_first=True)
    sum_h_qs = torch.sum(qs_hidden[0], dim=1)
    mean_pooled_h_qs = torch.div(sum_h_qs, torch.autograd.Variable(torch.FloatTensor(qs_seq_length)[:, np.newaxis]))
    
    qs_tuples = mean_pooled_h_qs.split(1+num_differing_questions)
    final_matrix_tuples_by_constituent_qs_by_hidden_size = torch.stack(qs_tuples, dim=0, out=None)

    return final_matrix_tuples_by_constituent_qs_by_hidden_size


'''Begin training'''

for epoch in range(num_epochs):
    for batch in range(1, num_batches+2):
        
        print("Working on batch #: ", batch)
        
        optimizer.zero_grad()
        questions_this_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]
        sequence_ids, dict_sequence_lengths = order_ids(questions_this_batch)

        print('got raw data')
        
        main_qs_tuples_matrix = construct_qs_matrix(questions_this_batch, dict_sequence_lengths, candidates=False)
        candidates_qs_tuples_matrix = construct_qs_matrix(sequence_ids, dict_sequence_lengths, candidates=True)

        print('got matrices')
        
        similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)

        print('got cosine similarity')
        
        target = Variable(torch.LongTensor([0] * int(len(sequence_ids)/(1+num_differing_questions))))
        loss_batch = loss_function(similarity_matrix, target)

        print('got loss of: ', loss_batch, ', calling backward soon!')
        
        loss_batch.backward()

        print('backward done, calling step soon')
        
        optimizer.step()
        
        print("loss on this batch: ", loss_batch.data[0])
