import math
import numpy as np
import time
from code.label_tree import LabelTree
from code.utils import *
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, vstack

class Parabel:
    '''
    Class that implements the Parabel technique for extreme multi-label classification. Parabel
    was originally designed by Prabhu et al. in their paper 'Parabel: Partitioned Label Trees for
    Extreme Classification with Application in Dynamic Search Advertising' (https://doi.org/10.1145/3178876.3185998)
    '''
    def __init__(self):
        '''Creates an instance of the Parabel class.'''
        self.tree = None
    

    def train(self, X, Y, max_labels_per_leaf, convert_X=True, outdir=None, verbose=True):
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

        :param convert_X: whether to convert the input from a list of strings to a matrix of numeric
        values.

        :param outdir: the directory (if any) where information regarding training time will be
        outputted.

        :param verbose: whether to print information about the status of the training process in the
        console as the procedure progresses.
        '''
        # Logging data to console.
        if verbose:
            print('Training has started.')
        train_start = time.time()

        # Vectorize inputs (if needed).
        if convert_X:
            if verbose:
                print('Vectorizing inputs...')
            start_time = time.time()
            X = vectorize_documents(X, 25000)
            duration = time.time() - start_time
            if verbose:
                print(f'Finished vectorizing inputs in {duration} seconds.')
            if outdir:
                save_keys_values_to_csv(outdir + 'times.csv', ['Vectorize inputs'], [duration])
        
        # Get the list of input indices associated with each label.
        labels_occurrences = get_labels_occurrences(Y)
        
        # Vectorize the labels.
        if verbose:
            print('Vectorizing labels...')
        start_time = time.time()
        labels_to_vectors_dict = vectorize_labels(X, labels_occurrences)
        duration = time.time() - start_time
        if verbose:
            print(f'Finished vectorizing labels in {duration} seconds.')
        if outdir:
            save_keys_values_to_csv(outdir + 'times.csv', ['Vectorize labels'], [duration])

        # Construct the label tree.
        if verbose:
            print('Constructing label tree...')
        start_time = time.time()
        self.tree = LabelTree(max_labels_per_leaf)
        self.tree.build(labels_to_vectors_dict)
        duration = time.time() - start_time
        if verbose:
            print(f'Finished building label tree in {duration} seconds.')
        if outdir:
            save_keys_values_to_csv(outdir + 'times.csv', ['Label tree build'], [duration])

        # For each internal node in the tree...
        for node in self.tree.internal_nodes:
            # Get the indices of the training points that are 'active' in this node, i.e. all the
            # training points that were tagged with labels contained in this node.
            active_indices = self._get_indices_of_inputs_active_in_node(node, labels_occurrences)
            
            # Iterate over the children of this node (left and right children).
            for child in node.get_children():
                # Get the indices of the training points that are 'active' and 'inactive' in
                # the child. The 'inactive' points are those that are active in the parent node
                # but not in the child.
                active_indices_child = self._get_indices_of_inputs_active_in_node(
                    child, labels_occurrences)
                inactive_indices_child = active_indices.difference(active_indices_child)
                
                # Get the positive and negative samples from the active and inactive indices.
                # Create the positive and negative labels for the samples.
                positive_samples = get_values_from_indices(X, active_indices_child)
                negative_samples = get_values_from_indices(X, inactive_indices_child)
                positive_labels = np.ones(len(positive_samples))
                negative_labels = np.zeros(len(negative_samples))

                # Join the positive and negative samples into one list. Do the same for labels.
                all_samples = vstack(positive_samples + negative_samples)
                all_labels = np.concatenate([positive_labels, negative_labels])

                # Fit the classifier with the data.
                child.fit_classifier(all_samples, all_labels)
        
        # For each leaf node in the tree...
        for leaf in self.tree.leaves:
            # Get the indices of the training points that are 'active' in this node.
            input_indices = self._get_indices_of_inputs_active_in_node(leaf, labels_occurrences)

            # For each label contained in this leaf node...
            for label in leaf.labels:
                # Get the positive samples (training points having the label) and negative samples
                # (training points belonging to the rest of the labels in this node).
                # Create the positive and negative labels for the samples.
                positive_samples = get_values_from_indices(X, labels_occurrences[label])
                inactive_indices = input_indices.difference(set(labels_occurrences[label]))
                negative_samples = get_values_from_indices(X, inactive_indices)
                positive_labels = np.ones(len(positive_samples))
                negative_labels = np.zeros(len(negative_samples))

                # Join the positive and negative samples into one list. Do the same for labels.
                all_samples = vstack(positive_samples + negative_samples)
                all_labels = np.concatenate([positive_labels, negative_labels])

                # Fit the classifier for the label with the data.
                leaf.fit_label_classifier(label, all_samples, all_labels)
        
        # Logging info to console and/or file.
        train_duration = time.time() - start_time
        if verbose:
            print(f'Finished training in {train_duration} seconds.')
        if outdir:
            save_keys_values_to_csv(outdir + 'times.csv', ['Train'], [train_duration])
    

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
            boundary_nodes_temp = boundary_nodes
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
            for label in leaf.labels:
                labels_probabilities[label] = leaf.predict_label_proba(label, x)
        
        # Sort the labels checked according to the probability of data point x being tagged with
        # them. Return the results.
        labels_sorted = sorted(labels_probabilities, key=labels_probabilities.get, reverse=True)
        return (labels_sorted, labels_probabilities)
    

    def evaluate(self, X, Y, search_width, metrics, metrics_args=None, outdir=None, verbose=True):
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

        :param outdir: the directory (if any) where information regarding evaluation time and scores
        will be outputted.

        :param verbose: whether to print information about the status of the evaluation process in
        the console as the procedure progresses.

        :returns: an array with the average scores obtained for each metric.
        '''
        # Logging info to the console and initializing variables.
        if verbose:
            print('Evaluation has started.')
        Y = [set(labels.split('\t')) for labels in Y]
        scores = []
        start_time = time.time()
        f1_index = None
        precisions_indices = dict()

        # Iterate over each test data point.
        size = X.shape[0] if type(X) is csr_matrix else len(X)
        for i in range(size):
            test_point = X[i]
            (labels_sorted, labels_probabilities) = self.predict(test_point, search_width)

            # Iterate over each metric.
            for j in range(len(metrics)):
                metric = metrics[j]
                metric_args = metrics_args[j]

                # Procedure for the F1_score_sample metric.
                if metric == ValidationMetric.F1_score_sample:
                    # Calculate the f1-score for the test point and accumulate it.
                    predicted = self._get_predicted_labels(labels_sorted, labels_probabilities)
                    predicted = set(predicted)
                    true_positives = len(Y[i].intersection(predicted))
                    precision = 0 if len(predicted) == 0 else true_positives / len(predicted)
                    recall = true_positives / len(Y[i])
                    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                    if f1_index is not None:
                        scores[f1_index] += f1
                    else:
                        scores.append(f1)
                        f1_index = len(scores) - 1

                # Procedure for the Precision_at_k metric.
                elif metric == ValidationMetric.Precision_at_k:
                    # Calculate the precision@k for the test point and accumulate it.
                    k = metric_args['k']
                    precision_at_k = 0
                    for l in range(min(k, len(labels_sorted))):
                        label = labels_sorted[l]
                        if label in Y[i]:
                            precision_at_k += 1
                    precision_at_k /= k
                    if k in precisions_indices:
                        scores[precisions_indices[k]] += precision_at_k
                    else:
                        scores.append(precision_at_k)
                        precisions_indices[k] = len(scores) - 1
        
        # Average the scores obtained over all test data points.
        scores = np.array(scores) / size

        # Logging info to console and/or file.
        duration = time.time() - start_time
        if verbose:
            print(f'Finished evaluation in {duration} seconds.')
            scores_str = [str(score) for score in scores]
            print(f'Scores obtained were: {" ".join(scores_str)}')
        if outdir:
            save_keys_values_to_csv(outdir + 'times.csv', ['Evaluate'], [duration])
            self._save_scores(metrics, scores, metrics_args, outdir)

        # Return scores.
        return scores
    

    def cross_validate(self, X, Y, folds_dict, max_labels_per_leaf=100, search_width=10, convert_X=True, metrics=[ValidationMetric.F1_score_sample], metrics_args=[None], best_metric_index=0, outdir=None, model_save_path=None, verbose=True):
        '''
        Performs a 10-fold cross validation of the Parabel classifier.

        :param X: matrix with shape (N, M), where N is the number of inputs and M is the number of
        features of each input. Can also be a list of 1D matrices. Can also be a list of strings, in
        which case 'convert_X' must be set to True.

        :param Y: an array of size N, where N is the number of input data points. Each position
        within the array has a string of labels (each one separated by a TAB character) that
        correspond to the labels with which the respective data point was tagged.

        :param folds_dict: a dictionary where the keys are the folds and the values are the lists of
        input indices that belong to them.

        :param max_labels_per_leaf: the maximum allowed number of labels to have in a leaf node of
        the label tree that will be generated.

        :param search_width: the maximum number of nodes that will be considered for checking for
        possible label assignment to the test points at each level of the tree. 

        :param metrics: list of ValidationMetrics to use for evaluating performance.

        :param metrics_args: list of dictionaries that is parallel to the metrics' list. Each
        dictionary contains parameters needed for the respective ValidationMetrics. If a given
        ValidationMetric does not require parameters, use None or an empty dictionary for it.

        :param best_metric_index: the index of the metric (within the metrics' list) to use for
        comparing models of different iterations of cross validation. Only used if 'model_save_path'
        is different from None, in which case it is used to determine which model (the best) to
        save.

        :param outdir: the directory (if any) where information regarding evaluation time and scores
        will be outputted.

        :param model_save_path: the path to the file where the best model will be saved.

        :param verbose: whether to print information about the status of the cross-validation
        process in the console as the procedure progresses.

        :returns: an array with the average cross validation scores obtained for each metric.
        '''
        # Logging info to the console.
        if verbose:
            print(f'Starting cross-validation...')

        # Vectorize inputs (if needed).
        if convert_X:
            if verbose:
                print('Vectorizing inputs...')
            start_time = time.time()
            X = vectorize_documents(X, 25000)
            duration = time.time() - start_time
            if verbose:
                print(f'Finished vectorizing inputs in {duration} seconds.')
            if outdir:
                save_keys_values_to_csv(outdir + 'times.csv', ['Vectorize inputs'], [duration])

        # Initialize values.
        start_time = time.time()
        scores = np.zeros(len(metrics))
        prev_best_score = -1

        # Perform 10 iterations varying the fold used for testing each time.
        for test_fold in range(10):
            # Logging info to the console.
            if verbose:
                print(f'Starting iteration {test_fold + 1} of cross-validation...')

            # Get the indices of the data that will be used for training and testing in this
            # iteration.
            train_indices = []
            for train_fold in range(len(folds_dict)):
                if train_fold != test_fold:
                    train_indices += folds_dict[train_fold]
            test_indices = folds_dict[test_fold]

            # Logging info to the console.
            if verbose:
                print(f'Getting training and testing inputs and outputs of this iteration...')

            # Get the actual inputs and labels from the train and test indices.
            train_inputs = get_values_from_indices(X, train_indices)
            test_inputs = get_values_from_indices(X, test_indices)
            train_labels = get_values_from_indices(Y, train_indices)
            test_labels = get_values_from_indices(Y, test_indices)
            
            # Train Parabel with the training data. Evaluate Parabel with the testing data.
            self.train(train_inputs, train_labels, max_labels_per_leaf,
                convert_X=False, outdir=outdir, verbose=verbose)
            current_score = self.evaluate(test_inputs, test_labels, search_width,
                metrics, metrics_args, outdir=outdir, verbose=True)
            scores += current_score
            
            # Check if the current model is the best model so far, in which case, save it.
            if model_save_path and current_score[best_metric_index] > prev_best_score:
                self.save_model(model_save_path)
                prev_best_score = current_score[best_metric_index]
        
        # Average the scores obtained in all the iterations.
        scores /= 10
        
        # Logging info to the console and/or file.
        duration = time.time() - start_time
        if verbose:
            print(f'Finished cross-validation in {duration} seconds.')
            scores_str = [str(score) for score in scores]
            print(f'Average scores obtained were: {" ".join(scores_str)}')
        if outdir:
            self._save_scores(metrics, scores, metrics_args, outdir, prefix='cross_val_')

        # Return scores.
        return scores
    

    def save_model(self, filePath, verbose=True):
        '''
        Saves the current Parabel object to a file.

        :param filePath: path to the file where the current object will be saved to.

        :param verbose: whether to print information regarding the save process.
        '''
        if verbose:
            print('Saving model...')
            start = time.time()
        dump(self, filePath)
        if verbose:
            duration = time.time() - start
            print(f'Successfully saved model to {filePath} in {duration} seconds.')
    

    @classmethod
    def load_model(cls, filePath, verbose=True):
        '''
        Loads a Parabel object from a file.

        :param filePath: path to the file where the Parabel object is stored.

        :param verbose: whether to print information regarding the loading process.

        :returns: the loaded Parabel object.
        '''
        if verbose:
            print(f'Loading model {filePath}...')
            start = time.time()
        model = load(filePath)
        if verbose:
            duration = time.time() - start
            print(f'Finished loading model {filePath} in {duration} seconds.')
        return model


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
    

    def _get_predicted_labels(self, labels_sorted, labels_probabilities):
        '''
        Gets the list of labels that were predicted for a data point.

        :param labels_sorted: list of predicted labels sorted from highest probability to lowest
        probability of being assigned to the data point.

        :param labels_probabilities: a dictionary, where the keys are the labels (strings) and the
        values are the probabilities of data point x being tagged with them.

        :returns: the list of labels that were predicted for a data point (i.e. had a probability
        greater than or equal to 0.5)
        '''
        left = 0
        right = len(labels_sorted) - 1
        while right - left > 1:
            mid = math.floor((left + right) / 2)
            label = labels_sorted[mid]
            if labels_probabilities[label] >= 0.5:
                left = mid + 1
            else:
                right = mid - 1
        if (labels_probabilities[labels_sorted[right]] >= 0.5):
            return labels_sorted[:right + 1]
        if (labels_probabilities[labels_sorted[left]] >= 0.5):
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
    
    
    def _save_scores(self, metrics, scores, metrics_args, outdir, prefix=''):
        '''
        Saves scores obtained by Parabel to disk.

        :param metrics: list of ValidationMetrics that were used for evaluating performance.

        :param scores: list of scores achieved by Parabel. This list is parallel to the metrics'
        list.

        :param metrics_args: list of dictionaries that is parallel to the metrics' list. Each
        dictionary contains parameters used within the respective ValidationMetrics. If a given
        ValidationMetric did not require parameters, None or an empty dictionary will be its value.

        :param outdir: directory where the scores are to be saved.

        :param prefix: string with which each validation metric is going to be prefixed when being
        saved to disk.
        '''
        metric_names = []
        for i in range(len(metrics)):
            metric_name = metrics[i].name
            if metric_name.find('k') != -1:
                k_value = metrics_args[i]['k']
                metric_name = metric_name.replace('k', str(k_value))
            metric_names.append(prefix + metric_name)
        save_keys_values_to_csv(outdir + 'scores.csv', metric_names, scores)