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

# Organize tools to fetch question test
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

# Cosine similarity distance metric
cosSim = CosineSimilarity()

# Loss function
loss_function = torch.nn.MarginRankingLoss(margin=0.2, size_average=False)

# Optimizer init, linking to model
weight_decay = 0.001
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01, weight_decay=0.001)


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

            # Get average hidden layer for q
            q_hidden = lstm(q_matrix_3d, (h0, c0))[0]
            avg_q_hidden = torch.sum(q_hidden, dim = 1) / q_matrix_3d.size()[1]

            # Get hidden layers for all p_plus and sum them for each sequence
            pos_q_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(pos_qs_padded, (h0, c0))[0], batch_first=True)
            sum_h_pos_q = torch.sum(pos_q_hidden[0], dim=1)

            # Get cosine similarity between q and all p_plus, average score and make reference vector for later usage
            score_pos_qs = Variable(torch.zeros(sum_h_pos_q.size()[0]).float())
            for i in range(sum_h_pos_q.size()[0]):
                avg_h_pos_q_i = (sum_h_pos_q[i] / pos_q_hidden[1][i]).unsqueeze(0)
                score_pos_q_i = cosSim.forward(avg_q_hidden, avg_h_pos_q_i)
                score_pos_qs[i] = score_pos_q_i

            score_pos_q_mean = score_pos_qs.mean()
            score_pos_q_vec = Variable(torch.ones(len(neg_qs_seq_length))).float() * score_pos_q_mean

            # Get hidden layers for all negative questions and sum them for each sequence
            neg_q_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(neg_qs_padded, (h0, c0))[0], batch_first=True)
            sum_h_neg_q = torch.sum(neg_q_hidden[0], dim = 1)

            # Average hidden layers for each p_minus and get cos similarity between each p_minus and q
            score_neg_qs = Variable(torch.zeros(sum_h_neg_q.size()[0]).float())
            for i in range(sum_h_neg_q.size()[0]):
                avg_h_neg_q_i = (sum_h_neg_q[i]/neg_q_hidden[1][i]).unsqueeze(0)
                score_neg_q_i = cosSim.forward(avg_q_hidden, avg_h_neg_q_i)
                score_neg_qs[i] = score_neg_q_i

            # Compare all the cos similarities (q,p_plus) v/s (q,p_minus) for all p_minus
            # Add loss for this question to batch loss which collects the losses for each q in the batch
            batch_loss += loss_function.forward(score_pos_q_vec, score_neg_qs, Variable(torch.ones(score_neg_qs.size()[0])))

        # Optimize model based on losses
        batch_loss.backward()
        optimizer.step()

        print("loss on this batch: ", batch_loss.data[0])

    # Save model after this epoch
    # Note: Model cannot be re-trained, but can be loaded for evaluation (See below)
    if pickle_each_epoch:
        torch.save(lstm, '../Pickled/LSTM_epoch' + str(epoch) + '.pt')

    # Run these line in evaluation scripts to load model:
    # lstm_loaded = torch.load('filename.pt')
    # lstm_loaded.eval()
