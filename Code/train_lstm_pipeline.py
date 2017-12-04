from preprocess import *
from scoring_metrics import *

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity

import time

'''Hyperparams dashboard'''
dropout = 0.1
margin = 0.2
lr = 10**-3

# Produces tensor [1 x num_words x input_size] for one particular question
def get_question_matrix(questionID, word2vec, id2Data, input_size):
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
    
    if num_words_found < 100:
        padding_rows = torch.zeros(100-num_words_found, input_size)
        q_matrix = torch.cat((q_matrix, padding_rows), 0)
    
    return [q_matrix.unsqueeze(0), num_words_found]


''' Data Prep '''
training_data = training_id_to_similar_different()
trainingQuestionIds = list(training_data.keys())
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData_truncate(100)

dev_data = devTest_id_to_similar_different(dev=True)
devQuestionIds = list(dev_data.keys())

''' Model Specs '''
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
bidirectional = False

lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function = torch.nn.MultiMarginLoss(margin=margin)
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

h0 = Variable(torch.zeros(1, 1, hidden_size), requires_grad=True)
c0 = Variable(torch.zeros(1, 1, hidden_size), requires_grad=True)


''' Procedural parameters '''
batch_size = 100
num_differing_questions = 20

num_epochs = 10
num_batches = round(len(trainingQuestionIds)/batch_size)


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
def order_ids(q_ids, dev=False):
    global training_data
    global dev_data
    global num_differing_questions
    
    if dev: data = dev_data
    else: data = training_data
        
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


'''Matrix constructors (use global vars, leave in order)'''
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
        q_matrix_3d, q_num_words = get_question_matrix(q, word2vec, id2Data, input_size)
        qs_matrix_list.append(q_matrix_3d)
        qs_seq_length.append(q_num_words)

    qs_padded = Variable(torch.cat(qs_matrix_list, 0))
    qs_hidden = lstm(qs_padded, (h0, c0)) # [ [num_q, num_word_per_q, hidden_size] i.e. all hidden, [1, num_q, hidden_size]  i.e. final hidden]
    sum_h_qs = torch.sum(qs_hidden[0], dim=1)
    mean_pooled_h_qs = torch.div(sum_h_qs, torch.autograd.Variable(torch.FloatTensor(qs_seq_length)[:, np.newaxis]))
    qs_tuples = mean_pooled_h_qs.split(1+num_differing_questions)
    final_matrix_tuples_by_constituent_qs_by_hidden_size = torch.stack(qs_tuples, dim=0, out=None)
    return final_matrix_tuples_by_constituent_qs_by_hidden_size


'''Begin training'''

for epoch in range(num_epochs):
    
    for batch in range(1, num_batches+1):
        start = time.time()
        
        print("Working on batch #: ", batch)
        
        optimizer.zero_grad()
        questions_this_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]
        sequence_ids, dict_sequence_lengths = order_ids(questions_this_batch)

        candidates_qs_tuples_matrix = construct_qs_matrix(sequence_ids, dict_sequence_lengths, candidates=True)
        main_qs_tuples_matrix = construct_qs_matrix(questions_this_batch, dict_sequence_lengths, candidates=False)
        
        similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)

        target = Variable(torch.LongTensor([0] * int(len(sequence_ids)/(1+num_differing_questions))))
        loss_batch = loss_function(similarity_matrix, target)

        loss_batch.backward()

        optimizer.step()
        
        print("loss_on_batch:", loss_batch.data[0], " time_on_batch:", time.time() - start)
        

    '''Dev eval after each epoch'''
        
    optimizer.zero_grad()
    sequence_ids, dict_sequence_lengths = order_ids(devQuestionIds, dev=True)

    candidates_qs_tuples_matrix = construct_qs_matrix(sequence_ids, dict_sequence_lengths, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix(devQuestionIds, dict_sequence_lengths, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)

    MRR_score = get_MRR_score(similarity_matrix)
    
    with open('logs.txt', 'a') as log_file:
        log_file.write('epoch: ' + str(epoch) + '\n')
        log_file.write('lr: ' + str(lr) +  ' marg: ' + str(margin) + ' drop: ' + str(dropout) + '\n' )        
        log_file.write('MRR: ' +  str(MRR_score) + '\n')

    print("MRR score on evaluation set:", MRR_score)

    target = Variable(torch.LongTensor([0] * int(len(sequence_ids)/(1+num_differing_questions))))
    loss_batch = loss_function(similarity_matrix, target)

    loss_batch.backward()

    optimizer.step()
    
    
    '''Save model for this epoch'''
    
    torch.save(lstm, '../Pickle/LSTM_m2d1l3epoch' + str(epoch) + '.pt')
