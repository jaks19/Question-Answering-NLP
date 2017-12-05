from preprocess import *
from scoring_metrics import *
from lstm_utils import *

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity

import time


'''Hyperparams dashboard'''
dropout = 0.1
margin = 0.2
lr = 10**-3


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


'''Begin training'''
for epoch in range(num_epochs):
    
    lstm.train()
    
    for batch in range(1, num_batches+1):
        start = time.time()
        print("Working on batch #: ", batch)
        
        optimizer.zero_grad()
        questions_this_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]
        sequence_ids, dict_sequence_lengths = organize_ids_training(questions_this_batch, training_data)

        candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, lstm, word2vec, id2Data, dict_sequence_lengths, candidates=True)
        main_qs_tuples_matrix = construct_qs_matrix_training(questions_this_batch, lstm, word2vec, id2Data, dict_sequence_lengths, candidates=False)
        similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)

        target = Variable(torch.LongTensor([0] * int(len(sequence_ids)/(1+num_differing_questions))))
        loss_batch = loss_function(similarity_matrix, target)

        loss_batch.backward()
        optimizer.step()
        print("loss_on_batch:", loss_batch.data[0], " time_on_batch:", time.time() - start)

        
    '''Dev eval after each epoch'''
    lstm.eval()
    dev_sequence_ids, p_pluses_indices_dict = organize_test_ids(devQuestionIds, dev_data)

    dev_candidates_qs_tuples_matrix = construct_qs_matrix_testing(dev_sequence_ids, lstm, word2vec, id2Data, candidates=True)
    dev_main_qs_tuples_matrix = construct_qs_matrix_testing(devQuestionIds, lstm, word2vec, id2Data, candidates=False)

    dev_similarity_matrix = torch.nn.functional.cosine_similarity(dev_candidates_qs_tuples_matrix, dev_main_qs_tuples_matrix, dim=2, eps=1e-08)
    MRR_score = get_MRR_score(dev_similarity_matrix, p_pluses_indices_dict)

    print("MRR score on dev set:", MRR_score)
    
    with open('logs.txt', 'a') as log_file:
        log_file.write('epoch: ' + str(epoch) + '\n')
        log_file.write('lr: ' + str(lr) +  ' marg: ' + str(margin) + ' drop: ' + str(dropout) + '\n' )        
        log_file.write('MRR: ' +  str(MRR_score) + '\n')

        
    '''Incorporate dev examples into training'''
    lstm.train()
    optimizer.zero_grad()
    sequence_ids, dict_sequence_lengths = organize_ids_training(devQuestionIds, dev_data)

    candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, lstm, word2vec, id2Data, dict_sequence_lengths, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_training(devQuestionIds, lstm, word2vec, id2Data, dict_sequence_lengths, candidates=False)
    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)

    target = Variable(torch.LongTensor([0] * int(len(sequence_ids)/(1+num_differing_questions))))
    loss_batch = loss_function(similarity_matrix, target)

    loss_batch.backward()
    optimizer.step()
    
    print("Trained on dev set")
    
    '''Save model for this epoch'''
    torch.save(lstm, '../Pickle/LSTM_m2d1l3epoch' + str(epoch) + '.pt')
