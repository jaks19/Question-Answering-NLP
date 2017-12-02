from preprocess import *
from get_q_matrices_functions import *

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity

''' Data Prep '''

# Organize training data
training_data = training_id_to_similar_different()
trainingQuestionIds = list(training_data.keys())

word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData()


''' Model Specs '''

# LSTM parameters
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 240
num_layers = 1
bias = True
batch_first = True
dropout = 0.2
bidirectional = False

# LSTM model, classifier, loss
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

# Loss function
loss_function = torch.nn.MarginRankingLoss(margin=0.2, size_average=False)

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

for epoch in range(num_epochs):
    for batch in range(1, num_batches+2):
        print("Working on batch # ", batch)

        # All question ids for this batch
        questions_this_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]

        # Init gradient and loss for this batch
        optimizer.zero_grad()
        batch_loss = Variable(torch.zeros(1).float())

        for q in questions_this_batch:
            # Build q's matrix
            q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(q, word2vec, id2Data)))

            # Grab positives and negatives, turn them into their respective matrices
            pos_qs = training_data[q][0]
            neg_qs = np.random.choice(training_data[q][1], num_differing_questions, replace = False)

            pos_qs_matrix_list = []
            pos_qs_seq_length = []
            for p_plus in pos_qs:
                pos_q_matrix_3d = get_question_matrix(p_plus, word2vec, id2Data)
                pos_qs_matrix_list.append(pos_q_matrix_3d)
                pos_qs_seq_length.append(pos_q_matrix_3d.shape[1])

            neg_qs_matrix_list = []
            neg_qs_seq_length = []
            for p_minus in neg_qs:
                neg_q_matrix_3d = get_question_matrix(p_minus, word2vec, id2Data)
                neg_qs_matrix_list.append(neg_q_matrix_3d)
                neg_qs_seq_length.append(neg_q_matrix_3d.shape[1])

            # Pad the positive and negative matrices and pack into a Packed_Sequence Object
            pos_qs_padded = padded_q_matrix(pos_qs_seq_length, pos_qs_matrix_list, input_size)
            neg_qs_padded = padded_q_matrix(neg_qs_seq_length, neg_qs_matrix_list, input_size)

            # Init hidden layer and cell state
            h0 = Variable(torch.zeros(1, 1, hidden_size))
            c0 = Variable(torch.zeros(1, 1, hidden_size))

            # Mean pooling hidden layer for q, normalized by num_words
            q_hidden = lstm(q_matrix_3d, (h0, c0))[0]
            avg_q_hidden = torch.sum(q_hidden, dim=1) / q_matrix_3d.size()[1]

            # Mean pooling hidden layer for each p_plus, normalized by num_words
            # Get score (q, p_plus) for each p_plus
            pos_q_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(pos_qs_padded, (h0, c0))[0], batch_first=True)
            sum_h_pos_q = torch.sum(pos_q_hidden[0], dim=1)
            score_pos_qs = Variable(torch.zeros(sum_h_pos_q.size()[0]).float())
            for i in range(sum_h_pos_q.size()[0]):
                avg_h_pos_q_i = (sum_h_pos_q[i] / pos_q_hidden[1][i]).unsqueeze(0)
                score_pos_q_i = cosSim.forward(avg_q_hidden, avg_h_pos_q_i)
                score_pos_qs[i] = score_pos_q_i

            # Mean pooling hidden layer for each p_minus, normalized by num_words
            # Get max_score_over_all_p_minus[ cos_sim (q, p_minus) ]
            neg_q_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(neg_qs_padded, (h0, c0))[0], batch_first=True)
            sum_h_neg_q = torch.sum(neg_q_hidden[0], dim=1)
            maxi_score = -100000
            for i in range(sum_h_neg_q.size()[0]):
                avg_h_neg_q_i = (sum_h_neg_q[i] / neg_q_hidden[1][i]).unsqueeze(0)
                score_neg_q_i = cosSim.forward(avg_q_hidden, avg_h_neg_q_i).data[0]
                if score_neg_q_i > maxi_score: maxi_score = score_neg_q_i
            maxi_score_tensor = Variable(torch.ones(1)) * maxi_score

            # For each (q,p_plus) pair, get loss by comparing cos_sim (q,p_plus) v/s max[(q,p_minus)] for all p_minus

            for score_q_p_plus in score_pos_qs:
                batch_loss += loss_function.forward(score_q_p_plus, maxi_score_tensor, Variable(torch.ones(1)))
        # Optimize model based on losses
        batch_loss.backward()
        optimizer.step()

        print("loss on this batch: ", batch_loss.data[0])


    ''' Pickle Model after each epoch'''

    # Note: Model cannot be re-trained, but can be loaded for evaluation (See below)
    if pickle_each_epoch:
        torch.save(lstm, '../Pickled/LSTM_final_epoch' + str(epoch) + '.pt')