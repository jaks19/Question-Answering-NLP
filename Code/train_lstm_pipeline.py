from preprocess import *
from scoring_metrics import *
from lstm_utils import *

import torch
from torch.autograd import Variable

import time

saved_model_name = "another_sunday_attempt"


'''Hyperparams dashboard'''
dropout = 0.2
margin = 0.2
lr = 10**-3


''' Data Prep '''
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData_truncate(100)

training_data = training_id_to_similar_different()
trainingQuestionIds = list(training_data.keys())

dev_data = devTest_id_to_similar_different(dev=True)
dev_question_ids = list(dev_data.keys())

test_data = devTest_id_to_similar_different(dev=False)
test_question_ids = list(test_data.keys())


''' Model Specs '''
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
bidirectional = True

lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function = torch.nn.MultiMarginLoss(margin=margin)
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

first_dim = num_layers * 2 if bidirectional else num_layers
h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Procedural parameters '''
batch_size = 100
num_differing_questions = 20
num_epochs = 25
num_batches = round(len(trainingQuestionIds)/batch_size)


def train_model(lstm, optimizer, batch_ids, batch_data, word2vec, id2Data):
    lstm.train()
    optimizer.zero_grad()

    sequence_ids, dict_sequence_lengths = organize_ids_training(batch_ids, batch_data, num_differing_questions)

    candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, lstm, h0, c0, word2vec, id2Data, dict_sequence_lengths, input_size, num_differing_questions, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_training(batch_ids, lstm, h0, c0, word2vec, id2Data, dict_sequence_lengths, input_size, num_differing_questions, candidates=False)
    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)

    target = Variable(torch.LongTensor([0] * int(len(sequence_ids)/(1+num_differing_questions))))
    loss_batch = loss_function(similarity_matrix, target)

    loss_batch.backward()
    optimizer.step()

    print("loss_on_batch:", loss_batch.data[0], " time_on_batch:", time.time() - start)
    return


def eval_model(lstm, ids, data, word2vec, id2Data):
    lstm.eval()
    sequence_ids, p_pluses_indices_dict = organize_test_ids(ids, data)

    candidates_qs_tuples_matrix = construct_qs_matrix_testing(sequence_ids, lstm, h0, c0, word2vec, id2Data, input_size, num_differing_questions, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_testing(ids, lstm, h0, c0, word2vec, id2Data, input_size, num_differing_questions, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)
    MRR_score = get_MRR_score(similarity_matrix, p_pluses_indices_dict)
    return MRR_score


'''Begin training'''
for epoch in range(num_epochs):

    # Train on whole training data set
    for batch in range(1, num_batches+1):
        start = time.time()
        questions_this_training_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]
        print("Working on batch #: ", batch)
        train_model(lstm, optimizer, questions_this_training_batch, training_data, word2vec, id2Data)
        
    # Evaluate on dev and test sets for MRR score
    dev_MRR_score = eval_model(lstm, dev_question_ids, dev_data, word2vec, id2Data)
    test_MRR_score = eval_model(lstm, test_question_ids, test_data, word2vec, id2Data)
    print("MRR score on dev set:", dev_MRR_score)
    print("MRR score on test set:", test_MRR_score)

    # Log results to local logs.txt file
    with open('logs.txt', 'a') as log_file:
        log_file.write('epoch: ' + str(epoch) + '\n')
        log_file.write('lr: ' + str(lr) +  ' marg: ' + str(margin) + ' drop: ' + str(dropout) + '\n' )        
        log_file.write('dev_MRR: ' +  str(dev_MRR_score) + '\n')
        log_file.write('test_MRR: ' +  str(test_MRR_score) + '\n')

    # Save model for this epoch
    torch.save(lstm, '../Pickle_lstm_part1/' + saved_model_name + '_epoch' + str(epoch) + '.pt')
