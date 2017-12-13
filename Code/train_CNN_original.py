from preprocess import *
from get_q_matrices_functions import *

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


''' Data Prep '''

# Organize training data
training_data = training_id_to_similar_different()
trainingQuestionIds = list(training_data.keys())

# Organize tools to fetch question test
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData()


''' Model Specs '''

# CNN parameters
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 100
kernel_size = 3
stride = 1
padding = 0
dilation = 1
groups = 1
bias = True

# CNN model
cnn = torch.nn.Sequential()
cnn.add_module('conv', torch.nn.Conv1d(in_channels = 200, out_channels = hidden_size, kernel_size = kernel_size, padding = padding, dilation = dilation, groups = groups, bias = bias).cuda())
cnn.add_module('tanh', torch.nn.Tanh().cuda())

# Loss function
loss_function = torch.nn.MarginRankingLoss(margin=0.2, size_average=False)

# Optimizer init, linking to model
weight_decay = 0.001
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01, weight_decay=0.001)


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
    epoch_loss = 0
    for batch in range(1, num_batches+2):
        print("Working on batch # ", batch)
        # All question ids for this batch
        questions_this_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]

        # Init gradient and loss for this batch
        optimizer.zero_grad()
        batch_loss = Variable(torch.zeros(1).cuda().float())

        for q in questions_this_batch:
            # Grab positives and negatives, turn them into their respective matrices
            pos_qs = training_data[q][0]
            neg_qs = np.random.choice(training_data[q][1], num_differing_questions, replace = False)
            
            
            # Build q matrix
            # Get hidden layer for q normalized by num_words
            q_matrix_3d = torch.transpose(Variable(torch.from_numpy(get_question_matrix(q, word2vec, id2Data)).cuda()), 1, 2)
            q_cnn_out = cnn(q_matrix_3d)
            normalized_q_hidden = (torch.sum(q_cnn_out, dim = 2) / q_matrix_3d.size()[2]).squeeze(0)
            
            
            # Build each p_plus matrix
            # Get hidden layer for each p_plus normalized by num_words
            # Get cosine similarity between each p_plus and q
            # Each pair is one basis of comparison against all the negatives
            num_hidden_layers_in_all = len(pos_qs)
            score_pos_qs = Variable(torch.zeros(num_hidden_layers_in_all).cuda().float())
            for i in range(len(pos_qs)):
                pos_q_matrix_3d = torch.transpose(Variable(torch.from_numpy(get_question_matrix(pos_qs[i], word2vec, id2Data)).cuda()), 1, 2)
                p_plus_cnn_out = cnn(pos_q_matrix_3d).squeeze(0)
                normalized_p_plus_hidden = (torch.sum(p_plus_cnn_out, dim = 1) / pos_q_matrix_3d.size()[2])
                score_pos_qs[i] = F.cosine_similarity(normalized_p_plus_hidden, normalized_q_hidden, dim = 0)
            
            
            # Build each p_minus matrix
            # Get hidden layer for each p_minus normalized by num_words
            # Get cosine similarity between each p_minus and q
            # Retrieve the max of all neg cos sims
            num_hidden_layers_in_all = len(neg_qs)
            maxi_score = Variable(torch.ones(1).float().cuda()) * -100000
            for i in range(len(neg_qs)):
                neg_q_matrix_3d = torch.transpose(Variable(torch.from_numpy(get_question_matrix(neg_qs[i], word2vec, id2Data)).cuda()), 1, 2)
                p_minus_cnn_out = cnn(neg_q_matrix_3d).squeeze(0)
                normalized_p_minus_hidden = (torch.sum(p_minus_cnn_out, dim = 1) / neg_q_matrix_3d.size()[2])
                score = F.cosine_similarity(normalized_p_minus_hidden, normalized_q_hidden, dim = 0)
                if score.data[0] > maxi_score.data[0]: 
                    maxi_score = score
            max_neg_cos_sim_variable = maxi_score
            
            
            # For each (q,p_plus) pair, get loss by comparing cos_sim (q,p_plus) v/s max[(q,p_minus)] for all p_minus
            for score_q_p_plus in score_pos_qs:
                batch_loss += loss_function.forward(score_q_p_plus,
                                                    max_neg_cos_sim_variable,
                                                    Variable(torch.ones(1).cuda()))

        # Optimize model based on losses
        batch_loss.backward()
        optimizer.step()

        print("loss on this batch: ", batch_loss.data[0])
        
    epoch_loss += batch_loss.data[0]

    print("loss on this epoch: ", epoch_loss)
    
    ''' Pickle_lstm_part1 Model after each epoch'''
    
    # Note: Model cannot be re-trained, but can be loaded for evaluation (See below)
    if pickle_each_epoch:
        torch.save(cnn, '../Pickle_lstm_part1/CNN_epoch' + str(epoch) + '.pt')


# Notes:
# Run these line in evaluation scripts to load model:
# lstm_loaded = torch.load('filename.pt')
# lstm_loaded.eval()