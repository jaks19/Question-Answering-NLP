from preprocess import *
from scoring_metrics import *
from cnn_utils import *


''' Data Prep '''
dev = True
testing_data = devTest_id_to_similar_different(dev)
testingQuestionIds = list(testing_data.keys())
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData_truncate(100)


''' Model (Specify pickled model name)'''
#cnn = torch.load('model file path')
cnn.eval()


'''Begin testing'''
sequence_ids, p_pluses_indices_dict = organize_test_ids(testingQuestionIds, testing_data)

candidates_qs_tuples_matrix = construct_qs_matrix_testing(sequence_ids, cnn, word2vec,
    id2Data, input_size, num_differing_questions, candidates=True)

main_qs_tuples_matrix = construct_qs_matrix_testing(testingQuestionIds, cnn, word2vec,
    id2Data, input_size, num_differing_questions, candidates=False)

similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix,
    dim=2, eps=1e-08)

MRR_score = get_MRR_score(similarity_matrix, p_pluses_indices_dict)

MRR_score = get_MRR_score(similarity_matrix, p_pluses_indices_dict)
MAP_score = get_MAP_score(similarity_matrix, p_pluses_indices_dict)
avg_prec_at_1 = avg_precision_at_k(similarity_matrix, p_pluses_indices_dict, 1)
avg_prec_at_5 = avg_precision_at_k(similarity_matrix, p_pluses_indices_dict, 5)

print("MRR score on test set:", MRR_score)
print("MAP score on test set:", MAP_score)
print("Precision at 1 score on test set:", avg_prec_at_1)
print("Precision at 5 score on test set:", avg_prec_at_5)