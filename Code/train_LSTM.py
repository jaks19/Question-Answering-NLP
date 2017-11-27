from Preprocess import get_words_and_embeddings
from Preprocess import questionID_to_questionData
from Preprocess import training_id_to_similar_different
from Preprocess import devTest_id_to_similar_different
from get_q_matrices_functions import get_question_matrix
from get_q_matrices_functions import padded_q_matrix

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity
import scipy.stats

# load/pre-process data
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData(py35 = True)
training_data = training_id_to_similar_different(py35 = True)
dev_data = devTest_id_to_similar_different(dev=True)
test_data = devTest_id_to_similar_different(dev=False)

# LSTM parameters
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
dropout = 0
bidirectional = False

# LSTM model
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

# initial hidden layer and cell state
h0 = Variable(torch.zeros(1, 1, 100))
c0 = Variable(torch.zeros(1, 1, 100))

# loss function
loss_function = torch.nn.MarginRankingLoss(margin=0.2, size_average=False)

# cosine similarity distance metric
cosSim = CosineSimilarity()

# number of negative examples to sample for each question
sample_size = 20

# number of questions before calling .backward()
batch_size = 100

# adam optimizer
weight_decay = 0.001
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01, weight_decay=0.001)

# epoch count and number of iterations through train data
epoch_number = 1
num_epochs = 1


# training with 100 questions per batch (100 postive questions and 100 sets of 20 negative questions)
for epoch in range(num_epochs):
	for batch in range(1, round(len(training_data.keys())/batch_size) + 2):
		# get questions for current batch
		try:
			q_batch = list(training_data.keys())[:batch_size*batch]
		except IndexError:
			q_batch = list(training_data.keys())[batch_size*(batch-1):]
        
		# zeroing gradient and batch_loss
		optimizer.zero_grad()
		batch_loss = Variable(torch.zeros(1).float())
		count = 0
		for q in q_batch:
		# for tracking
			print(count)
			count += 1
            
			# get question matrix
			q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(q, word2vec, id2Data)))

			# get positive question matrix
			pos_q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(training_data[q][0][0], word2vec, id2Data)))

			# get negative question IDs and convert questions to matrices
			neg_qs = np.random.choice(training_data[q][1], sample_size, replace = False)
			neg_qs_matrix_list = []
			neg_qs_seq_length = []
			# for p in neg_qs:
				neg_q_matrix_3d = get_question_matrix(p, word2vec, id2Data)
				neg_qs_matrix_list.append(neg_q_matrix_3d)
				neg_qs_seq_length.append(neg_q_matrix_3d.shape[1])

			# do padding for negative questions
			neg_qs_padded = padded_q_matrix(neg_qs_seq_length, neg_qs_matrix_list, input_size)

			# initial hidden layer and cell state
			h0 = Variable(torch.zeros(1, 1, 100))
			c0 = Variable(torch.zeros(1, 1, 100))

			# get hidden layers for question and average them
			q_hidden = lstm(q_matrix_3d, (h0, c0))[0]
			avg_h_q = torch.sum(q_hidden, dim = 1)/q_matrix_3d.size()[1]

			# get hidden layers for positive question and average them
			pos_q_hidden = lstm(pos_q_matrix_3d, (h0, c0))[0]
			avg_h_pos_q = torch.sum(pos_q_hidden, dim = 1)/pos_q_matrix_3d.size()[1]

			# get cosine similarity between question and positive question; repeat for tensor length 20 for loss function input
			score_pos_q = cosSim.forward(avg_h_q, avg_h_pos_q)
			score_pos_q_vec = Variable(torch.ones(len(neg_qs_seq_length))).float() * score_pos_q

			# get hidden layers for all negative questions and sum then for each sequence
			neg_q_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(neg_qs_padded, (h0, c0))[0], batch_first=True)
			sum_h_neg_q = torch.sum(neg_q_hidden[0], dim = 1)

			# average hidden layers for each negative question and get cosine similarities between negative questions and positive question
			score_neg_qs = Variable(torch.zeros(sum_h_neg_q.size()[0]).float())
			for i in range(sum_h_neg_q.size()[0]):
				avg_h_neg_q_i = (sum_h_neg_q[i]/neg_q_hidden[1][i]).unsqueeze(0)
				score_neg_q_i = cosSim.forward(avg_h_q, avg_h_neg_q_i)
				score_neg_qs[i] = score_neg_q_i

			# add loss for question to batch loss
			batch_loss += loss_function.forward(score_pos_q_vec, score_neg_qs, Variable(torch.ones(score_neg_qs.size()[0])))

		# take gradient wrt batch_loss
		batch_loss.backward()
        
		print(batch_loss.data[0])

		# update weights
		optimizer.step()

        
        
# training with 1 questions per batch (1 postive questions and 1 set of 20 negative questions)     
'''
for epoch in range(num_epochs):
	for q in list(training_data.keys()):
		# zeroing gradient
		optimizer.zero_grad()
        
		# get question matrix
		q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(q, word2vec, id2Data)))

		# get positive question matrix
		pos_q_matrix_3d = Variable(torch.from_numpy(get_question_matrix(training_data[q][0][0], word2vec, id2Data)))

		# get negative question IDs and convert questions to matrices
		neg_qs = np.random.choice(training_data[q][1], sample_size, replace = False)
		neg_qs_matrix_list = []
		neg_qs_seq_length = []
		for p in neg_qs:
			neg_q_matrix_3d = get_question_matrix(p, word2vec, id2Data)
			neg_qs_matrix_list.append(neg_q_matrix_3d)
			neg_qs_seq_length.append(neg_q_matrix_3d.shape[1])

		# do padding for negative questions
		neg_qs_padded = padded_q_matrix(neg_qs_seq_length, neg_qs_matrix_list, input_size)

		# get hidden layers for question and average them
		q_hidden = lstm(q_matrix_3d, (h0, c0))[0]
		avg_h_q = torch.sum(q_hidden, dim = 1)/q_matrix_3d.size()[1]
		
		# get hidden layers for positive question and average them
		pos_q_hidden = lstm(pos_q_matrix_3d, (h0, c0))[0]
		avg_h_pos_q = torch.sum(pos_q_hidden, dim = 1)/pos_q_matrix_3d.size()[1]

		# get cosine similarity between question and positive question; repeat for tensor length 20 for loss function input
		score_pos_q = cosSim.forward(avg_h_q, avg_h_pos_q)
		score_pos_q_vec = Variable(torch.ones(len(neg_qs_seq_length))).float() * score_pos_q

		# get hidden layers for all negative questions and sum then for each sequence
		neg_q_hidden = torch.nn.utils.rnn.pad_packed_sequence(lstm(neg_qs_padded, (h0, c0))[0], batch_first=True)
		sum_h_neg_q = torch.sum(neg_q_hidden[0], dim = 1)

		# average hidden layers for each negative question and get cosine similarities between negative questions and positive question
		score_neg_qs = Variable(torch.zeros(sum_h_neg_q.size()[0]).float())
		for i in range(sum_h_neg_q.size()[0]):
			avg_h_neg_q_i = (sum_h_neg_q[i]/neg_q_hidden[1][i]).unsqueeze(0)
			score_neg_q_i = cosSim.forward(avg_h_q, avg_h_neg_q_i)
			score_neg_qs[i] = score_neg_q_i

		# add loss for question to batch loss
		batch_loss = loss_function.forward(score_pos_q_vec, score_neg_qs, Variable(torch.ones(score_neg_qs.size()[0])))

		# take gradient wrt batch_loss
		batch_loss.backward()
        
		print(batch_loss.data[0])

		# update weights
		optimizer.step()
'''



