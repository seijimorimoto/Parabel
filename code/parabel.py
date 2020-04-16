import math
import numpy as np
from label_tree import LabelTree
from sklearn.linear_model import LogisticRegression
from utils import get_labels_occurrences, vectorize_documents, vectorize_labels, ValidationMetric

class Parabel:
    '''
    Class that implements the Parabel technique for extreme multi-label classification. Parabel
    was originally designed by Prabhu et al. in their paper 'Parabel: Partitioned Label Trees for
    Extreme Classification with Application in Dynamic Search Advertising' (https://doi.org/10.1145/3178876.3185998)
    '''
    def __init__(self):
        '''Creates an instance of the Parabel class.'''
        self.tree = None
    
    def train(self, X, Y, max_labels_per_leaf, convert_X=True):
        '''
        Executes the training process of the Parabel technique.

        :param X: matrix with shape (N, M), where N is the number of inputs and M is the number of
        features of each input. Can also be a list of 1D matrices. Can also be a list of strings, in
        which case 'convert_X' must be set to True.

        :param Y: an array of size N, where N is the number of input data points. Each position
        within the array has a string of labels (each one separated by a TAB character) that
        correspond to the labels with which the respective data point was tagged.

        :param max_labels_per_leaf: the maximum allowed number of labels to have in a leaf node of
        the label tree that will be generated.

        :convert_X: whether to convert the input from a list of strings to a matrix of numeric
        values.
        '''
        # Vectorize inputs (if needed) and labels.
        input_matrix = vectorize_documents(X, 25000) if convert_X else X
        labels_occurrences = get_labels_occurrences(Y)
        labels_to_vectors_dict = vectorize_labels(input_matrix, labels_occurrences)

        # Construct the label tree.
        self.tree = LabelTree(max_labels_per_leaf)
        self.tree.build(labels_to_vectors_dict)

        # For each internal node in the tree...
        for node in self.tree.internal_nodes:
            # Get the indices of the training points that are 'active' in this node, i.e., all the
            # training points that were tagged with labels contained in this node.
            active_indices = self._get_indices_of_inputs_active_in_node(node, labels_occurrences)
            
            # Iterate over the children of this node (left and right children).
            for child in node.get_children():
                if child not in self.tree.leaves:
                    # Get the indices of the training points that are 'active' and 'inactive' in
                    # the child. The 'inactive' points are those that are active in the parent node
                    # but not in the child. 
                    active_indices_child = self._get_indices_of_inputs_active_in_node(child, labels_occurrences)
                    inactive_indices_child = active_indices.difference(active_indices_child)
                    
                    # Get the positive and negative samples from the active and inactive indices.
                    # Create the positive and negative labels for the samples.
                    positive_samples = self._get_inputs_from_indices(X, active_indices_child)
                    negative_samples = self._get_inputs_from_indices(X, inactive_indices_child)
                    positive_labels = np.ones(len(positive_samples))
                    negative_labels = np.zeros(len(negative_samples))

                    # Join the positive and negative samples into one list. Do the same for labels.
                    all_samples = positive_samples + negative_samples
                    all_labels = np.concatenate([positive_labels, negative_labels])

                    # Fit the classifier with the data.
                    child.classifier.fit(all_samples, all_labels)
        
        # For each leaf node in the tree...
        for leaf in self.tree.leaves:
            # Get the indices of the training points that are 'active' in this node.
            input_indices = self._get_indices_of_inputs_active_in_node(leaf, labels_occurrences)

            # For each label contained in this leaf node...
            for label in leaf.labels:
                # Get the positive samples (training points having the label) and negative samples
                # (training points belonging to the rest of the labels in this node).
                # Create the positive and negative labels for the samples.
                positive_samples = self._get_inputs_from_indices(X, labels_occurrences[label])
                inactive_indices = input_indices.difference(set(labels_occurrences[label]))
                negative_samples = self._get_inputs_from_indices(X, inactive_indices)
                positive_labels = np.ones(len(positive_samples))
                negative_labels = np.zeros(len(negative_samples))

                # Join the positive and negative samples into one list. Do the same for labels.
                all_samples = positive_samples + negative_samples
                all_labels = np.concatenate([positive_labels, negative_labels])

                # Fit the classifier with the data.
                leaf.labels_classifiers[label] = LogisticRegression(
                    fit_intercept=False, solver='liblinear')
                leaf.labels_classifiers[label].fit(all_samples, all_labels)
    
    def predict(self, x, search_width):
        '''
        Predicts the labels that correspond to a given data point.

        :param x: the data point for which the labels are going to be predicted.

        :param search_width: the maximum number of nodes that will be considered for checking for
        possible label assignment to the data point at each level of the tree. 

        :returns: a two-element tuple. The first element is the list of labels checked sorted in
        descending order, such that the first label is the one with the most probability of being
        assigned to the data point. The second element is a dictionary, where the keys are the
        labels (strings) and the values are the probabilities of data point x being tagged with
        them. 
        '''
        depth = self.tree.depth
        root = self.tree.root
        boundary_nodes = set([root])

        # Explore all internal nodes level by level.
        for _ in range(depth - 1):
            boundary_nodes_temp = boundary_nodes.copy()
            boundary_nodes = set()
            # For each node in the boundary nodes of the current level...
            for node in boundary_nodes_temp:
                # Calculate the log-likelihood of data point x belonging to the probability
                # distribution of each child node. Add the log-likelihood of the parent to each
                # child so that the total probability of the data point x belonging to the path
                # from the root node to the current nodes is obtained.
                for child in node.get_children():
                    child.log_likelihood = child.predict_log_proba(x) + node.log_likelihood
                    boundary_nodes.add(child)
            # Retain only the nodes with the greatest log-likelihood.
            boundary_nodes = self._retain_most_probable_nodes(boundary_nodes, search_width)
        
        # For each label classifier in each leaf node reached through the search process, calculate
        # the probability of the data point being tagged with the label.
        labels_probabilities = dict()
        for leaf in boundary_nodes:
            for label, classifier in leaf.labels_classifiers.items():
                # We use the second position in the tuple returned by predict_proba, because the
                # first one corresponds to the probability of being in class 0 (rejected).
                labels_probabilities[label] = classifier.predict_proba(x)[1]
        
        # Sort the labels checked according to the probability of data point x being tagged with
        # them. Return the results.
        labels_sorted = sorted(labels_probabilities, key=labels_probabilities.get, reverse=True)
        return (labels_sorted, labels_probabilities)
    
    def evaluate(self, X, Y, search_width, metrics, metrics_args=None):
        '''
        Evaluates the performance of the Parabel classifier on a test set.

        :param X: matrix with shape (N, M), where N is the number of test points and M is the
        number of features of each point. Can also be a list of 1D matrices.

        :param Y: an array of size N, where N is the number of test points. Each position within
        the array has a string of labels (each one separated by a TAB character) that correspond to
        the labels with which the respective test point was tagged.

        :param search_width: the maximum number of nodes that will be considered for checking for
        possible label assignment to the test points at each level of the tree. 

        :param metrics: list of ValidationMetrics to use for evaluating performance.

        :param metrics_args: list of dictionaries that is parallel to the metrics' list. Each
        dictionary contains parameters needed for the respective ValidationMetrics. If a given
        ValidationMetric does not require parameters, use None or an empty dictionary for it.

        :returns: an array with the average scores obtained for each metric.
        '''
        Y = [set(labels.split('\t')) for labels in Y]
        scores = []

        # Iterate over each metric.
        for i in range(len(metrics)):
            metric = metrics[i]
            metric_args = metrics_args[i]

            # Procedure for the F1_score_sample metric.
            if metric == ValidationMetric.F1_score_sample:
                f1_total = 0
                # Calculate the f1-score for each test point. Accumulate all the results and average
                # them.
                for i in range(len(X)):
                    test_point = X[i]
                    (labels_sorted, labels_probabilities) = self.predict(test_point, search_width)
                    predicted = self._get_predicted_labels(labels_sorted, labels_probabilities)
                    predicted = set(predicted)
                    true_positives = len(Y[i].intersection(predicted))
                    precision = true_positives / len(predicted)
                    recall = true_positives / len(Y[i])
                    f1 = 2 * precision * recall / (precision + recall)
                    f1_total += f1
                f1_total /= len(X)
                scores.append(f1_total)

            # Procedure for the Precision_at_k metric.
            elif metric == ValidationMetric.Precision_at_k:
                precision_at_k_total = 0
                # Calculate the precision@k for each test point. Accumulate all the results and
                # average them.
                for i in range(len(X)):
                    test_point = X[i]
                    k = metric_args['k']
                    (labels_sorted, _) = self.predict(test_point, search_width)
                    precision_at_k = 0
                    for i in range(min(k, len(labels_sorted))):
                        label = labels_sorted[i]
                        if label in Y[i]:
                            precision_at_k += 1
                    precision_at_k /= k
                    precision_at_k_total += precision_at_k
                precision_at_k_total /= len(X)
                scores.append(precision_at_k_total)
        
        # Return scores.
        return np.array(scores)
    
    def cross_validate(self, X, Y, folds, max_labels_per_leaf=100, search_width=10, convert_X=True, metrics=[ValidationMetric.F1_score_sample], metrics_args=[None]):
        '''
        Performs a 10-fold cross validation of the Parabel classifier.

        :param X: matrix with shape (N, M), where N is the number of inputs and M is the number of
        features of each input. Can also be a list of 1D matrices. Can also be a list of strings, in
        which case 'convert_X' must be set to True.

        :param Y: an array of size N, where N is the number of input data points. Each position
        within the array has a string of labels (each one separated by a TAB character) that
        correspond to the labels with which the respective data point was tagged.

        :param folds: an array of size N, where N is the number of input data points. Each value
        within the array must indicate the fold (0-9) to which the corresponding input data point
        belongs. Values greater than 9 will be considered extra data that will be used as part of
        the training set in each iteration of the 10-fold cross validation.

        :param max_labels_per_leaf: the maximum allowed number of labels to have in a leaf node of
        the label tree that will be generated.

        :param search_width: the maximum number of nodes that will be considered for checking for
        possible label assignment to the test points at each level of the tree. 

        :param metrics: list of ValidationMetrics to use for evaluating performance.

        :param metrics_args: list of dictionaries that is parallel to the metrics' list. Each
        dictionary contains parameters needed for the respective ValidationMetrics. If a given
        ValidationMetric does not require parameters, use None or an empty dictionary for it.

        :returns: an array with the average cross validation scores obtained for each metric.
        '''
        # Vectorize inputs (if needed), get a mapping of folds to the input indices and initialize
        # the scores to zero.
        input_matrix = vectorize_documents(X, 25000) if convert_X else X
        folds_dict = self._get_folds_dictionary(folds)
        scores = np.zeros(len(metrics))

        # Perform 10 iterations varying the fold used for testing each time.
        for test_fold in range(10):
            # Get the indices of the data that will be used for training and testing in this
            # iteration.
            train_indices = []
            for train_fold in range(len(folds_dict)):
                if train_fold != test_fold:
                    train_indices += folds_dict[train_fold]
            test_indices = folds_dict[test_fold]

            # Get the actual inputs and labels from the train and test indices.
            train_inputs = self._get_inputs_from_indices(input_matrix, train_indices)
            test_inputs = self._get_inputs_from_indices(input_matrix, test_indices)
            train_labels = self._get_inputs_from_indices(Y, train_indices)
            test_labels = self._get_inputs_from_indices(Y, test_indices)
            
            # Train Parabel with the training data. Evaluate Parabel with the testing data. 
            self.train(train_inputs, train_labels, max_labels_per_leaf, convert_X=False)
            scores += self.evaluate(test_inputs, test_labels, search_width, metrics, metrics_args)
        
        # Average the scores obtained in all the iterations.
        scores /= 10
        return scores
    
    def _get_folds_dictionary(self, folds):
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

    def _get_indices_of_inputs_active_in_node(self, node, labels_occurrences):
        '''
        Gets the indices of the inputs that were tagged with labels that are contained within a node
        of the label tree.

        :param node: the node for which the inputs' indices are to be found.

        :param labels_ocurrences: a dictionary where the keys are labels (strings). The value
        associated with each key/label is the list of indices of the inputs that were tagged with
        that label.

        :returns: a set of indices of the inputs active in the node.
        '''
        x_indices = set()
        for label in node.labels:
            for index in labels_occurrences[label]:
                x_indices.add(index)
        return x_indices
    
    def _get_inputs_from_indices(self, X, indices):
        '''
        Gets the inputs that are placed in specific row indices of the input matrix.

        :param X: matrix with shape (N, M), where N is the number of inputs and M is the number of
        features of each input.

        :param indices: set of indices indicating the rows that will be extracted from X and
        returned in this method.

        :returns: a list of vectors representing the inputs located at the given positions within
        the matrix X.
        '''
        inputs = []
        for index in indices:
            inputs.append(X[index])
        return inputs
    
    def _get_predicted_labels(self, labels_sorted, labels_probabilities):
        '''
        Gets the list of labels that were predicted for a data point.

        :param labels_sorted: list of predicted labels sorted from highest probability to lowest
        probability of being assigned to the data point.

        :param labels_probabilities: list of probabilities of the predicted labels being assigned to
        the data point. It is parallel to labels_sorted.

        :returns: the list of labels that were predicted for a data point (i.e. had a probability
        greater than or equal to 0.5)
        '''
        left = 0
        right = len(labels_sorted) - 1
        while right - left > 1:
            mid = math.floor((left + right) / 2)
            if labels_probabilities[mid] >= 0.5:
                left = mid + 1
            else:
                right = mid - 1
        if (labels_probabilities[right] >= 0.5):
            return labels_sorted[:right + 1]
        if (labels_probabilities[left] >= 0.5):
            return labels_sorted[:left + 1]
        return labels_sorted[:left]
    
    def _retain_most_probable_nodes(self, nodes, retain_size):
        '''
        Retains the nodes that have the highest log-likelihoods.

        :param nodes: sets of nodes from which to select the ones with highest log-likelihoods.

        :param retain_size: how many nodes to retain.

        :returns: the set of the nodes with the highest log-likelihoods.
        '''
        sorted_nodes = sorted(nodes, key = lambda node : node.log_likelihood, reverse=True)
        return set(sorted_nodes[:retain_size])