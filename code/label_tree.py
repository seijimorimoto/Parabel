import math
import numpy as np
import random
from sklearn.linear_model import LogisticRegression


class LabelNode:
    '''
    An internal or leaf node of a LabelTree.
    '''

    def __init__(self):
        '''
        Creates a node.
        '''
        self.classifier = LogisticRegression(fit_intercept=False, solver='liblinear')
        self.labels_classifiers = dict()
        self.labels = None
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.log_likelihood = 0
        pass

    def get_children(self):
        '''Returns the list of the left and right children.'''
        return [self.left_child, self.right_child]

    def set_left_child(self, node):
        '''
        Sets the left child of this node.

        :param node: the node that will be the left child of this node.
        '''
        self.left_child = node
        node.parent = self

    def set_right_child(self, node):
        '''
        Sets the right child of this node.

        :param node: the node that will be the right child of this node.
        '''
        self.right_child = node
        node.parent = self
    
    def predict_log_proba(self, x):
        '''
        Predict the log-likelihood of point x belonging to the probability distribution of this
        node's classifier. Note that this method should only be used for internal nodes.

        :param x: the data point for which the log-likelihood is to be calculated.

        :returns: the log-likelihood. 
        '''
        return self.classifier.predict_log_proba([x])


class LabelTree:
    '''
    A binary tree data structure for hierarchically grouping labels into clusters. 
    '''

    def __init__(self, max_labels_per_leaf=100):
        '''
        Creates the object that will hold the tree.

        :param max_labels_per_leaf: the maximum allowed number of labels to have in a leaf node.
        '''
        self.depth = 0
        self.internal_nodes = set()
        self.leaves = set()
        self.root = LabelNode()
        self.max_labels_per_leaf = max_labels_per_leaf

    def build(self, labels_to_vectors_dict):
        '''
        Builds the tree to hold the labels stored in the labels_to_vector_dict object.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.
        '''
        self.depth = math.ceil(math.log2(len(labels_to_vectors_dict) / self.max_labels_per_leaf))
        self.root.labels = set(labels_to_vectors_dict.keys())
        self.internal_nodes.add(self.root)
        self._grow_node_recursive(self.root, labels_to_vectors_dict)

    def _grow_node_recursive(self, node, labels_to_vectors_dict):
        '''
        Grow a node/subtree by recursively splitting the labels into two clusters (one for each
        child) until leaf nodes are reached.

        :param node: the node that represents the root of the subtree from which labels will be
        recursively partitioned into clusters.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.
        '''
        # Check if the node is a leaf, in which case terminate the recursion.
        if len(node.labels) <= self.max_labels_per_leaf:
            self.leaves.add(node)
        else:
            # Create left and right children of node.
            n_left = LabelNode()
            n_right = LabelNode()

            # Partition the labels so that half goes to the left child and the rest to right one.
            (labelset_left, labelset_right) = self._partition(node.labels, labels_to_vectors_dict)

            # Assign the labels to the children nodes.
            n_left.labels = labelset_left
            n_right.labels = labelset_right
            node.set_left_child(n_left)
            node.set_right_child(n_right)
            self.internal_nodes.add(n_left)
            self.internal_nodes.add(n_right)

            # Recursively grow the left and right children nodes.
            self._grow_node_recursive(n_left, labels_to_vectors_dict)
            self._grow_node_recursive(n_right, labels_to_vectors_dict)

    def _partition(self, node_labels, labels_to_vectors_dict):
        '''
        Partitions a set of labels into two sets (roughly equivalent in size). This is an
        implementation of the spherical balanced k-means algorithm, for k=2.

        :param node_labels: set of labels (strings) to partition.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.

        :returns: a tuple where the first element is the set of labels assigned to the left
        partition and the second element is the set of labels assigned to the right one.
        '''
        # Initialize a mean vector for the left and right partitions by uniformly sampling from all
        # the label vectors.
        mean_vector_left = random.choice(labels_to_vectors_dict.values())
        mean_vector_right = random.choice(labels_to_vectors_dict.values())

        # Initialize the sets of labels that will be assigned to the left and right partitions.
        labelset_left = set()
        labelset_right = set()
        labelset_left_prev = set()
        labelset_right_prev = set()

        # Iterate over this loop which adjusts the label assignments to the left and right
        # partitions until there is no change in assignments between consecutive iterations.
        while labelset_left != labelset_left_prev or labelset_right != labelset_right_prev:
            # Get the similarities of the labels to the left and right partitions.
            (node_labels_ordered, similarities) = self._get_cluster_similarities(
                mean_vector_left, mean_vector_right, node_labels, labels_to_vectors_dict)

            # Set the current sets of labels assigned to the left and right partitions to be the
            # previous sets. The current sets are now emptied so that new assignments are made in
            # this iteration.
            labelset_left_prev = labelset_left
            labelset_right_prev = labelset_right
            labelset_left = set()
            labelset_right = set()

            # Get the dictionary mapping the labels to their rankings according to the similarities
            # to the left and right partitions.
            labels_rankings = self._get_labels_rankings(
                node_labels_ordered, similarities)

            # Iterate over the labels to partition. The labels that have a low ranking will be
            # assigned to the left partition. The labels with a high ranking will be assigned to the
            # right partition. The label exactly in the middle will be asssigned to a partition
            # based on the sign of the similarity-value of the label to the left and right clusters.
            for i in range(len(node_labels_ordered)):
                label = node_labels_ordered[i]
                rank = labels_rankings[label]
                if rank < math.ceil(len(node_labels) / 2):
                    labelset_left.add(label)
                elif rank > math.ceil(len(node_labels) / 2):
                    labelset_right.add(label)
                else:
                    if similarities[i] >= 0:
                        labelset_left.add(label)
                    else:
                        labelset_right.add(label)

            # Update the mean vectors of the left and right partitions based on the labels that were
            # that were assigned to them in this iteration.
            mean_vector_left = self._update_mean_vector(labelset_left, labels_to_vectors_dict)
            mean_vector_right = self._update_mean_vector(labelset_right, labels_to_vectors_dict)

        # Return sets of labels for the left and right partitions.
        return (labelset_left, labelset_right)

    def _get_cluster_similarities(self, mean_vector_left, mean_vector_right, node_labels, labels_to_vectors_dict):
        '''
        Compute the 'similarities' between label vectors and the mean vectors of left and right
        partitions/clusters.

        :param mean_vector_left: the mean vector of the left partition.

        :param mean_vector_right: the mean vector of the right partition.

        :param node_labels: the set of labels (strings) for which the similarities to the left and
        right partitions will be computed.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.

        :returns: a tuple where its first element is a list of the labels (strings) for which the
        similarities were computed (essentially, this is the same as the node_labels, but in the
        form of a list instead of a set). The second element is a list with the similarity scores
        of the labels (this list is parallel to the labels' list).
        '''
        # Initialize lists and calculate the transpose of the mean vectors of the partitions.
        similarities = []
        node_labels_ordered = []
        left_t = mean_vector_left.transpose()
        right_t = mean_vector_right.transpose()

        # For each label, obtain its vector representation and compute the similarities.
        for label in node_labels:
            label_vector = labels_to_vectors_dict[label]
            similarities.append((left_t.dot(label_vector) - right_t.dot(label_vector))[0][0])
            node_labels_ordered.append(label)

        # Return the lists.
        return (node_labels_ordered, similarities)

    def _get_labels_rankings(self, node_labels_ordered, values):
        '''
        Ranks labels by sorting them in descending order according to some values.

        :param node_labels_ordered: list of labels to rank (ordered in the name of this parameter
        just means that it is parallel to the values array).

        :param values: list of values used for ordering the labels. Each value within this list is
        associated with the label in the same index in the node_labels_ordered list.

        :returns: a dictionary where the keys are the labels (strings) and the values correspond to
        their ranks (each rank will be a value between 0 (inclusive) and the length of
        node_labels_ordered (exclusive)).
        '''
        # Get the list of sorted indices in descending order.
        values = np.array(values)
        values = -1 * values
        sorted_indices = np.argsort(values)
        labels_rankings = dict()

        # Rank the labels based on the sorted indices.
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]
            label = node_labels_ordered[index]
            labels_rankings[label] = i
        return labels_rankings

    def _update_mean_vector(self, labelset, labels_to_vectors_dict):
        '''
        Computes a mean label vector from a set of labels.

        :param labelset: set of labels for which a mean vector is to be obtained.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.

        :returns: the mean label vector.
        '''
        # Add the vector representations of the labels in the labelset.
        updated_mean = 0
        for label in labelset:
            updated_mean += labels_to_vectors_dict[label]

        # Divide the vector of sums by its euclidean norm to obtain the mean vector.
        euclidean_norm = updated_mean.dot(
            updated_mean.transpose()).toarray()[0][0]
        euclidean_norm = math.sqrt(euclidean_norm)
        updated_mean = updated_mean / euclidean_norm
        return updated_mean
