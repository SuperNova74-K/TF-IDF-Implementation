import math
import string
import numpy as np

def _dictionarize_documents(documents):
    documents_frequencies = []
    for i, document in enumerate(documents):
        dictionary = {}
        for word in document:
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1
        documents_frequencies.append(dictionary)
    return documents_frequencies


def _preprocess_documents(documents):
    preprocessed_documents = []

    for i, document in enumerate(documents):
        # lowercase every letter
        document = document.lower()

        # remove punctuation
        document = document.translate(str.maketrans('', '', string.punctuation))

        # tokenize
        document = document.split(' ')
        preprocessed_documents.append(document)

    return preprocessed_documents


class TF_IDF:
    def __init__(self, documents):
        self._documents = documents

        self._documents = _preprocess_documents(self._documents)

        self._documents_frequencies = _dictionarize_documents(self._documents)

        self._vocab = set()
        for document in self._documents:
            self._vocab.update(document)
        self._vocab = sorted(list(self._vocab))

    def tf(self, word, document_index):
        return self._documents_frequencies[document_index].get(word, 0)

    def idf(self, word):
        nt = sum([(1 if word in document else 0) for document in self._documents_frequencies])
        return math.log(
            ((1 + len(self._documents)) / (1 + nt))
        , math.e) + 1


    def tfidf(self, word, document_index):
        return self.tf(word, document_index) * self.idf(word)

    def tfidf_all_words(self, document_index, normalized:bool ,limit=None):
        word_scores = [(word, self.tfidf(word, document_index)) for word in self._documents[document_index]]

        scores = [word_score[1] for word_score in word_scores]

        if normalized:
            norm = np.linalg.norm(scores)
            scores = scores / norm

        indexes = np.argsort(scores)[::-1]
        word_scores = [word_scores[i] for i in indexes]

        if limit:
            word_scores = word_scores[:limit]

        return word_scores

    def word_embedding(self, word):
        return np.array([self.tfidf(word=word, document_index=i) for i in range(len(self._documents))])

    def document_embedding(self, document_index, normalized:bool):
        document = np.array([self.tfidf(word=word, document_index=document_index) for word in self._vocab])
        if normalized:
            norm = np.linalg.norm(document)
            document = document / norm
        return document