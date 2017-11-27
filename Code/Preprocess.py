# Word mapped to its embedding vector
def get_words_and_embeddings():
    filepath = "../Data1/vectors_pruned.200.txt"
    lines = open(filepath).readlines()
    word2vec = {}
    for line in lines:
        word_coordinates_list = line.split(" ")
        word2vec[word_coordinates_list[0]] = word_coordinates_list[1:-1]
    return word2vec


# Question id mapped to the content: title + body text
def questionID_to_questionData(py35 = False):
    filepath = "../Data1/text_tokenized.txt"
    if py35 == True:
        lines = open(filepath, encoding = 'utf8').readlines()
    elif py35 == False:
        lines = open(filepath).readlines()

    id2Data = {}
    for line in lines:
        id_title_body_list = line.split('\t')
        id2Data[int(id_title_body_list[0])] = id_title_body_list[1].split(" ") + id_title_body_list[2].split(" ")
    return id2Data


# For training data, question id mapped to [[similar_questions_ids], [different_questions_ids]]
def training_id_to_similar_different(py35 = False):
    filepath = "../Data1/train_random.txt"
    if py35 == True:
        lines = open(filepath, encoding = 'utf8').readlines()
    elif py35 == False:
        lines = open(filepath).readlines()

    training_data = {}
    for line in lines:
        id_similarids_diffids = line.split('\t')
        question_id = int(id_similarids_diffids[0])
        similar_ids = id_similarids_diffids[1].split(" ")
        different_ids = id_similarids_diffids[2].split(" ")

        for i in range(len(similar_ids)): similar_ids[i] = int(similar_ids[i])
        for j in range(len(different_ids)): different_ids[j] = int(different_ids[j])

        training_data[question_id] = [ similar_ids, different_ids ]
    return training_data


# For test data, question id mapped to [[similar_questions_ids], [different_questions_ids]]
# For dev set, use dev=True, for test set use dev=False
# Note:
#   If there are 20 similar questions, then the list of different questions ids is empty [ [...], [] ]
#   Also, there might be no similar question for an id and all are different [ [], [...] ]
def devTest_id_to_similar_different(dev=True):
    filepath = "../Data1/dev.txt" if dev else "../Data1/test.txt"
    lines = open(filepath).readlines()

    evaluation_data = {}
    for line in lines:
        id_similarids_diffids = line.split('\t')
        question_id = int(id_similarids_diffids[0])
        similar_ids = id_similarids_diffids[1].split(" ") if id_similarids_diffids[1] != '' else []
        different_ids = id_similarids_diffids[2].split(" ")
        for i in range(len(similar_ids)): similar_ids[i] = int(similar_ids[i])
        for j in range(len(different_ids)): different_ids[j] = int(different_ids[j])
        different_ids = list(set(different_ids) - set(similar_ids))
        evaluation_data[question_id] = [similar_ids, different_ids]
    return evaluation_data


# word2vec = get_words_and_embeddings()

# id2Data = questionID_to_questionData()

# training_data = training_id_to_similar_different()

# dev_data = devTest_id_to_similar_different(dev=True)

# test_data = devTest_id_to_similar_different(dev=False)