from preprocess import *
from get_q_matrices_functions import *

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity


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










''' Data Prep '''

# Organize training data
training_data = training_id_to_similar_different()
trainingQuestionIds = list(training_data.keys())

word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData()


''' Model Specs '''

# LSTM parameters
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
dropout = 0.2
bidirectional = False

# LSTM model, classifier, loss
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

# Loss function
loss_function = torch.nn.MultiMarginLoss(margin=0.2)

# Optimizer init, linking to model
optimizer = torch.optim.Adam(lstm.parameters(), lr=10**-4, weight_decay=0.001)

# Cos-similatity
cosSim = CosineSimilarity()

''' Procedural parameters '''

# Sampling numbers
batch_size = 100
num_differing_questions = 20

# How much to train?
num_epochs = 10
num_batches = round(len(training_data.keys())/batch_size)

# Save model after each epoch?
pickle_each_epoch = True

''' Begin Training '''
h0 = Variable(torch.zeros(1, 1, hidden_size), requires_grad=True)
c0 = Variable(torch.zeros(1, 1, hidden_size), requires_grad=True)
    
for epoch in range(num_epochs):
    for batch in range(1, num_batches+2):
        
        batch_loss = []
        print("Working on batch # ", batch)
        
        # Init gradient and loss for this batch
        optimizer.zero_grad()
        
        # All question ids for this batch
        questions_this_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]
        
        total_loss = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
        
        for q in questions_this_batch:
            
            # Mean pooled hidden layer for q
            q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(q, word2vec, id2Data)))
            q_hidden = lstm(q_matrix_3d, (h0, c0))[0]
            avg_q_hidden = torch.sum(q_hidden, dim=1) / q_matrix_3d.size()[1]

            
            
            
            # Space left because I had refactored this into the method implemented on top of this page,
            # I moved code back in when I saw the no graph node error...
            
            
            pos_qs = training_data[q][0]
            num_pos = len(pos_qs)
            
            qs_matrix_list = []
            qs_seq_length = []

            for q_pos in pos_qs:
                q_matrix_3d = get_question_matrix(q_pos, word2vec, id2Data)
                qs_matrix_list.append(q_matrix_3d)
                qs_seq_length.append(q_matrix_3d.shape[1])

            qs_padded = padded_q_matrix(qs_seq_length, qs_matrix_list, input_size)
            qs_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(qs_padded, (h0, c0))[0], batch_first=True)
            sum_h_qs = torch.sum(qs_hidden[0], dim=1)

            lst_hidden = []
            for i in range(len(sum_h_qs)): 
                h = sum_h_qs[i] / qs_seq_length[i]
                lst_hidden.append(h)
        
            mean_pooled_hidden_layers_pos = list(lst_hidden)
            
            

            
            # Again, Space left because I had refactored this into the method implemented on top of this page,
            # I moved code back in when I saw the no graph node error...
            
            neg_qs = np.random.choice(training_data[q][1], num_differing_questions, replace = False)
            num_neg = len(neg_qs)
            
            qs_matrix_list = []
            qs_seq_length = []

            for q_neg in neg_qs:
                q_matrix_3d = get_question_matrix(q_neg, word2vec, id2Data)
                qs_matrix_list.append(q_matrix_3d)
                qs_seq_length.append(q_matrix_3d.shape[1])

            qs_padded = padded_q_matrix(qs_seq_length, qs_matrix_list, input_size)
            qs_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(qs_padded, (h0, c0))[0], batch_first=True)
            sum_h_qs = torch.sum(qs_hidden[0], dim=1)

            lst_hidden = []
            for i in range(len(sum_h_qs)): 
                h = sum_h_qs[i] / qs_seq_length[i]
                lst_hidden.append(h)
            
            mean_pooled_hidden_layers_neg = list(mean_pooled_hidden_layers_for_ids(neg_qs, input_size))
            
            
            
            
            
            
            
            
            
            
            
            # A tuple is (q+, q-, q--, q--- ...)
            # Matrix of all tuples (num_tuples x num_qs_in_tuple x hidden_size)
            
            matrix_all_tuples = Variable(torch.FloatTensor(num_pos, 1+num_neg, hidden_size))
            for i, h_pos in enumerate(mean_pooled_hidden_layers_pos):
                this_tuple = torch.stack([h_pos]+mean_pooled_hidden_layers_neg)
                matrix_all_tuples[i] = this_tuple
            
            matrix_q_only = Variable(torch.FloatTensor(num_pos, 1+num_neg, hidden_size))
            for i in range(len(mean_pooled_hidden_layers_pos)):
                this_tuple = torch.stack([avg_q_hidden]*(1+num_neg))
                matrix_q_only[i] = this_tuple
            
            
            similarity_matrix = torch.nn.functional.cosine_similarity(matrix_all_tuples, matrix_q_only, dim=2, eps=1e-08)
            target = Variable(torch.LongTensor([0]*num_pos))
            mini_batch_loss = loss_function(similarity_matrix, target)
            
            batch_loss.append(mini_batch_loss)

        total_loss = torch.cat(batch_loss).sum()
        total_loss.backward()
        optimizer.step()

        print("loss on this batch: ", total_loss.data[0])
