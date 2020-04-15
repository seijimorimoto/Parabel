import numpy as np
from label_tree import LabelTree
from sklearn.linear_model import LogisticRegression

class Parabel:
    '''
    Class that implements the Parabel technique for extreme multi-label classification. Parabel
    was originally designed by Prabhu et al. in their paper 'Parabel: Partitioned Label Trees for
    Extreme Classification with Application in Dynamic Search Advertising' (https://doi.org/10.1145/3178876.3185998)
    '''
    def __init__(self):
        '''Creates an instance of the Parabel class.'''
        self.tree = None
    
    def train(self, X, Y, max_labels_per_leaf, labels_occurrences, labels_to_vectors_dict):
        '''
        Executes the training process of the Parabel technique.

        :param X: matrix with shape (N, M), where N is the number of inputs and M is the number of
        features of each input.
        '''
        # TODO: Update method comment when the exact function parameters are known.
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
    
    def _retain_most_probable_nodes(self, nodes, retain_size):
        '''
        Retains the nodes that have the highest log-likelihoods.

        :param nodes: sets of nodes from which to select the ones with highest log-likelihoods.

        :param retain_size: how many nodes to retain.

        :returns: the set of the nodes with the highest log-likelihoods.
        '''
        sorted_nodes = sorted(nodes, key = lambda node : node.log_likelihood, reverse=True)
        return set(sorted_nodes[:retain_size])