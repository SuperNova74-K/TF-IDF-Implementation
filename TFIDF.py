import math
import string


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

    def tf(self, word, document_index):
        return self._documents_frequencies[document_index].get(word, 0)

    def idf(self, word):
        nt = sum([(1 if word in document else 0) for document in self._documents_frequencies])
        return math.log(
            ((1 + len(self._documents)) / (1 + nt))
        , math.e) + 1


    def tfidf(self, word, document_index):
        return self.tf(word, document_index) * self.idf(word)
