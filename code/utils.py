import math
import nltk
import pandas as pd
import re
from enum import Enum
from nltk.stem import WordNetLemmatizer
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


class ValidationMetric(Enum):
    '''Enum that holds the different metrics available for evaluating the Parabel technique'''
    F1_score_sample = 1
    Precision_at_k = 2


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


def get_folds_dictionary(folds):
    '''
    Gets a mapping of folds to the training indices belonging to them.

    :param folds: an array of size N, where N is the number of input data points. Each value
    within the array must indicate the fold to which the corresponding input data point belongs.

    :returns: a dictionary where the keys are the folds and the values are the lists of input
    indices that belong to them.
    '''
    folds_dict = dict()
    for i in range(len(folds)):
        fold = folds[i]
        if fold not in folds_dict:
            folds_dict[fold] = [i]
        else:
            folds_dict[fold] += [i]
    return folds_dict


def get_labels_occurrences(Y):
    '''
    :param Y: an array of size N, where N is the number of input data points. Each position within
    the array has a string of labels (each one separated by a TAB character) that correspond to the
    labels with which the respective data point was tagged.

    :returns: a dictionary where the keys are labels (strings). The value associated with each key /
    label is the list of indices of the inputs that were tagged with that label.
    '''
    labels_occurrences = dict()
    index = 0
    for labels in Y:
        for label in labels.split('\t'):
            if label not in labels_occurrences:
                labels_occurrences[label] = [index]
            else:
                labels_occurrences[label] += [index]
        index += 1
    return labels_occurrences


def get_values_from_indices(collection, indices):
    '''
    Gets the values that are placed in specific row indices of a collection (matrix or list).

    :param collection: collection (matrix or list) from which items / values will be extracted.

    :param indices: set of indices indicating the rows that will be extracted from collection and
    returned in this method.

    :returns: a list of items representing the values located at the given positions within the
    collection.
    '''
    values = []
    for index in indices:
        values.append(collection[index])
    return values


def save_keys_values_to_csv(path, keys, values):
    '''
    Saves key-value pairs within the same line of a CSV file.

    :param path: the file path to the CSV file where the key-value pairs will be saved.

    :param keys: the list of keys to save in the CSV file.

    :param values: the list of values to save in the CSV file.
    '''
    with open(path, 'a') as f:
        line = ''
        for i in range(len(keys)):
            line = line + f',{keys[i]},{values[i]}'
        line = line[1:]
        f.write(f'{line}\n')


def vectorize_documents(X, max_features=25000):
    '''
    Turns a series of text documents into Bag-of-Words (BoW) TF-IDF vectors.

    :param X: an array of size N, where N is the number of input data points. Each position within
    the array is a document as a string.

    :param max_features: maximum number of features to keep in the BoW TF-IDF vectors.

    :returns: a matrix of size (N, max_features), meaning all the documents as BoW TF-IDF vectors.
    '''
    vectorizer = TfidfVectorizer(
        preprocessor=Preprocessor(), tokenizer=LemmaTokenizer(), max_features=max_features)
    return vectorizer.fit_transform(X)


def vectorize_labels(input_matrix, labels_occurrences):
    '''
    Turns a series of labels into their vector notations based on the input data points that were
    tagged with the labels.

    :param input_matrix: csr_matrix with shape (N, M), where N is the number of inputs and M is the
    number of features of each input.

    :param labels_occurrences: a dictionary where the keys are labels (strings). The value linked
    with each key/label is the list of indices of the inputs that were tagged with that label.

    :returns: a dictionary containing the labels (strings) as its keys. The value associated with each key is the vector representation of the label.
    '''
    labels = list(labels_occurrences.keys())
    labels_to_vectors_dict = dict()

    # Iterate over the labels.
    for label in labels:
        labels_to_vectors_dict[label] = 0
        # Add all the input vectors that were tagged with the current label.
        for input_index in labels_occurrences[label]:
            labels_to_vectors_dict[label] += input_matrix[input_index]
        # Divide the vector of sums by its euclidean norm to obtain the mean vector, representing
        # the label.
        vector = labels_to_vectors_dict[label]
        euclidean_norm = vector.dot(vector.transpose()).toarray()[0][0]
        euclidean_norm = math.sqrt(euclidean_norm)
        if euclidean_norm != 0:
            labels_to_vectors_dict[label] /= euclidean_norm
    
    return labels_to_vectors_dict