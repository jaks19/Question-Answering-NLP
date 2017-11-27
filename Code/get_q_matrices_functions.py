import numpy as np
import torch
from torch.autograd import Variable
import scipy.stats

# function takes a question ID, the word2vec dictionary, and the id2Data dictionary
# returns a 3d matrix that is 1 x sequence_length x word_vec_length
# when wrapped in a pytorch variable, matrices produced by this function can be fed to the LSTM
def get_question_matrix(questionID, word2vec, id2Data):
    q_word_vecs = []
    for word in id2Data[questionID]:
        try:
            word_vec = np.array(word2vec[word]).astype(np.float32).reshape(len(word2vec[word]), -1)
            q_word_vecs.append(word_vec)
        except KeyError:
            pass
        
    q_matrix = np.concatenate(q_word_vecs, axis = 1).T
    q_matrix_3d = q_matrix.reshape(-1, q_matrix.shape[0], q_matrix.shape[1])
    
    return q_matrix_3d


# function takes a list of sequence lengths and a list of matrices size 1 x sequence_length x word_vec_length
# returns a pytorch packed sequence object that can be fed to the LSTM
def padded_q_matrix(q_seq_lengths, q_matrix_list, input_size):
    max_q_seq_length = np.max(q_seq_lengths)
    q_matrix = np.zeros((len(q_seq_lengths), max_q_seq_length, input_size))
    q_matrix_loc = scipy.stats.rankdata(-np.array(q_seq_lengths), method = 'ordinal') - 1
    
    for i in range(len(q_seq_lengths)):
        q_matrix[q_matrix_loc[i], :q_seq_lengths[i]] = q_matrix_list[i]
    
    q_matrix = Variable(torch.from_numpy(q_matrix)).float()
    q_padded = torch.nn.utils.rnn.pack_padded_sequence(q_matrix, np.sort(q_seq_lengths)[::-1], batch_first=True)
    
    return q_padded