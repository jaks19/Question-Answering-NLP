import numpy as np
import torch
from torch.autograd import Variable
import scipy.stats

def get_question_matrix(questionID, word2vec, id2Data):
    # Get the vector representation for each word in this question as list [v1,v2,v3,...]
    q_word_vecs = []
    for word in id2Data[questionID]:
        try:
            word_vec = np.array(word2vec[word]).astype(np.float32).reshape(len(word2vec[word]), -1)
            q_word_vecs.append(word_vec)
        except KeyError:
            pass

    # Make it into a 3D array with the first axis representing a batch size of 1 here
    # When batching for training/testing, the first dimension is vital
    q_matrix = np.concatenate(q_word_vecs, axis=1).T
    q_matrix_3d = q_matrix.reshape(-1, q_matrix.shape[0], q_matrix.shape[1])
    return q_matrix_3d


# Function organizes a batch of data
# Takes list of question lengths and corresponding question matrices of size: 1 x question_length x word_dimension
# Returns Pytorch Packed_Sequence object (batch that can be fed to the LSTM)
def padded_q_matrix(q_seq_lengths, q_matrix_list, input_size):
    # Get longest question length
    max_q_seq_length = np.max(q_seq_lengths)

    # Initialize empty Matrix: batch_size x max_question_length x word_dimension
    q_matrix = np.zeros((len(q_seq_lengths), max_q_seq_length, input_size))

    # Order questions in descending order of num_words and place into batch
    q_matrix_loc = scipy.stats.rankdata(-np.array(q_seq_lengths), method = 'ordinal') - 1
    for i in range(len(q_seq_lengths)):
        q_matrix[q_matrix_loc[i], :q_seq_lengths[i]] = q_matrix_list[i]
    q_matrix = Variable(torch.from_numpy(q_matrix)).float().cuda()

    # Pack and pad batch
    q_padded = torch.nn.utils.rnn.pack_padded_sequence(q_matrix, np.sort(q_seq_lengths)[::-1], batch_first=True)
    return q_padded
