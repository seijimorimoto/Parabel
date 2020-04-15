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
                child.classifier.fit(all_samples, all_labels)
    
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