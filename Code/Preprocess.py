# 1. Organize the Embedding Matrix and Word2Index Dict
'''
In ../word_vectors.txt, we have a word, a space, then space-separated floats until a newline character
The word's embedding is the vector next to it and we need to separate those unique words from their vectors and build:
- A dict of word to index
- A matrix of each of these vectors in order so that row i is the ith word's embedding vector
'''
def get_words_and_embeddings():
    filepath = "../Data1/vectors_pruned.200.txt"
    lines = open(filepath).readlines()
    word2vec = {}
    for line in lines:
        word_coordinates_list = line.split(" ")
        word2vec[word_coordinates_list[0]] = word_coordinates_list[1:-1]
    return word2vec

word2vec = get_words_and_embeddings()
print word2vec['zero']
print len(word2vec['zero'])
