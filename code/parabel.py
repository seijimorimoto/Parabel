import nltk
import numpy as np
import os
import pandas as pd
import re
import time
from joblib import dump, load
from nltk.stem import WordNetLemmatizer
from math import sqrt
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class LemmaTokenizer:
    '''
    Callable class for tokenizing a document. The tokens will be the lemmatized words.
    Lemmatization is carried out using the morphological processing of WordNet.
    '''

    def __init__(self):
        '''Constructs the callable object'''
        self.wnl = WordNetLemmatizer()

    def __call__(self, document):
        '''
        Calls the object for tokenizing a document.

        :param document: the document to tokenize.
        '''
        return [self.wnl.lemmatize(word) for word in nltk.word_tokenize(document)]


class Preprocessor:
    '''
    Callable class for carrying out pre-processing tasks on a document.
    '''

    def __call__(self, document):
        '''
        Calls the object for pre-processing a document. Preprocessing consists of joining words that
        have a hyphen, discarding all characters except for sequences of alphabetic characters with
        a length of two or more, and lower-casing the document.

        :param document: the document to pre-process.
        '''
        document = document.split('-')
        document = ''.join(document)
        pattern = re.compile('[a-zA-Z]{2,}')
        document = pattern.findall(document)
        document = ' '.join(document)
        document = document.lower()
        return document


def load_dataset(path):
    '''
    Loads a dataset.

    :param path: Path to the dataset to load.
    '''
    return pd.read_csv(path)


def prepare_nltk():
    '''Prepares NLTK by downloading specific packages.'''
    nltk.download('punkt')
    nltk.download('wordnet')


def transform_labels_to_vectors(input_matrix, labels_ocurrences, labels_to_vectors_dict):
    '''
    Transforms labels to their vector representation by averaging the input vectors that were tagged
    with each label.

    :param input_matrix: csr_matrix with shape (N, M), where N is the number of inputs and M is the
    number of features of each input.

    :param labels_ocurrences: a dictionary where the keys are labels (strings). The value associated
    with each key/label is the list of indices of the inputs that were tagged with that label.

    :param labels_to_vectors_dict: a dictionary containing the labels (strings) as its keys. The
    value associated with each key (which will be the vector representation of the label) will be
    populated once this method finishes executing. 
    '''
    # Iterate over the labels.
    for label, occurrences in labels_ocurrences.items():
        # Add all the input vectors that were tagged with the current label.
        for index in occurrences:
            labels_to_vectors_dict[label] += input_matrix[index]
        # Divide the vector of sums by its euclidean norm to obtain the mean vector, representing
        # the label.
        vector = labels_to_vectors_dict[label]
        euclidean_norm = vector.dot(vector.transpose()).toarray()[0][0]
        euclidean_norm = sqrt(euclidean_norm)
        labels_to_vectors_dict[label] /= euclidean_norm


def main():
    prepare_nltk()
    if (os.path.exists('data/econbiz/econbiz_inputs_vectors.joblib')):
        Y = pd.read_pickle('data/econbiz/econbiz_labels.pkl')
        folds = pd.read_pickle('data/econbiz/econbiz_folds.pkl')
        input_matrix = load('data/econbiz/econbiz_inputs_vectors.joblib')
    else:
        vectorizer = TfidfVectorizer(
            preprocessor=Preprocessor(), tokenizer=LemmaTokenizer(), max_features=25000)
        print('Loading dataset...')
        t = time.time()
        dataset = load_dataset('data/econbiz/econbiz.csv')
        print(f'Finish loading dataset in {time.time() - t} seconds')
        X = dataset['title']
        Y = dataset['labels']
        folds = dataset['fold']
        pd.to_pickle(Y, 'data/econbiz/econbiz_labels.pkl')
        pd.to_pickle(folds, 'data/econbiz/econbiz_folds.pkl')
        print('Vectorizing the inputs...')
        t = time.time()
        input_matrix = vectorizer.fit_transform(X)
        print(f'Finish vectorizing inputs in {time.time() - t} seconds')
        print(input_matrix.shape)
        dump(input_matrix, 'data/econbiz/econbiz_inputs_vectors.joblib')

    if (os.path.exists('data/econbiz/econbiz_labels_vectors.joblib')):
        labels_to_vectors_dict = load(
            'data/econbiz/econbiz_labels_vectors.joblib')
    else:
        labels_to_vectors_dict = dict()
        labels_occurrences = dict()

        index = 0
        for labels in Y:
            for label in labels.split('\t'):
                if label not in labels_to_vectors_dict:
                    labels_to_vectors_dict[label] = csr_matrix((1, 25000))
                    labels_occurrences[label] = [index]
                else:
                    labels_occurrences[label] += [index]
            index += 1

        print(f'Vectorizing the labels...')
        t = time.time()
        transform_labels_to_vectors(
            input_matrix, labels_occurrences, labels_to_vectors_dict)
        print(f'Finish vectorizing labels in {time.time() - t} seconds')
        dump(labels_to_vectors_dict, 'data/econbiz/econbiz_labels_vectors.joblib')


if __name__ == "__main__":
    main()
