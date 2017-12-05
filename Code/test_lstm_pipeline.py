from preprocess import *
from scoring_metrics import *
from lstm_utils import *

from torch.nn.modules.distance import CosineSimilarity

import time


''' Data Prep '''
dev = True
testing_data = devTest_id_to_similar_different(dev)
testingQuestionIds = list(testing_data.keys())[:10]
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData_truncate(100)


''' Model (Specify pickled model name)'''
lstm = torch.load('../Pickle/LSTM_m2d1l3epoch1.pt')
lstm.eval()


'''Begin testing'''
sequence_ids, p_pluses_indices_dict = organize_test_ids(testingQuestionIds, testing_data)

candidates_qs_tuples_matrix = construct_qs_matrix_testing(sequence_ids, lstm, word2vec, id2Data, candidates=True)
main_qs_tuples_matrix = construct_qs_matrix_testing(testingQuestionIds, lstm, word2vec, id2Data, candidates=False)

similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)
MRR_score = get_MRR_score(similarity_matrix, p_pluses_indices_dict)

if dev: print("MRR score on dev set:", MRR_score)
else: print("MRR score on test set:", MRR_score)