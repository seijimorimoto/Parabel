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
        self.classifier = LogisticRegression(dual=True, solver='liblinear', max_iter=20)
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
        return self.classifier.predict_log_proba(x)[0][1]


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
        self.root.labels = set(labels_to_vectors_dict.keys())
        self._grow_node_recursive(self.root, labels_to_vectors_dict)

    def _grow_node_recursive(self, node, labels_to_vectors_dict, level=0):
        '''
        Grow a node/subtree by recursively splitting the labels into two clusters (one for each
        child) until leaf nodes are reached.

        :param node: the node that represents the root of the subtree from which labels will be
        recursively partitioned into clusters.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.

        :param level: the level within the tree in which the node is located.
        '''
        # Check if the node is a leaf, in which case terminate the recursion.
        if len(node.labels) <= self.max_labels_per_leaf:
            self.leaves.add(node)
            if level + 1 > self.depth:
                self.depth = level + 1
        else:
            # Create left and right children of node.
            self.internal_nodes.add(node)
            n_left = LabelNode()
            n_right = LabelNode()

            # Partition the labels so that half goes to the left child and the rest to right one.
            (labelset_left, labelset_right) = self._partition(node.labels, labels_to_vectors_dict)

            # Assign the labels to the children nodes.
            n_left.labels = labelset_left
            n_right.labels = labelset_right
            node.set_left_child(n_left)
            node.set_right_child(n_right)

            # Recursively grow the left and right children nodes.
            self._grow_node_recursive(n_left, labels_to_vectors_dict, level + 1)
            self._grow_node_recursive(n_right, labels_to_vectors_dict, level + 1)

    def _partition(self, node_labels, labels_to_vectors_dict, epsilon=0.0001):
        '''
        Partitions a set of labels into two sets (roughly equivalent in size). This is an
        implementation of the spherical balanced k-means algorithm, for k=2.

        :param node_labels: set of labels (strings) to partition.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.

        :param epsilon: float value that specifies the threshold for terminating the partitioning
        procedure. If in consecutive iterations of the partitioning algorithm, the objective
        function (sum of similarities of all labels to their respective assigned partitions) does
        not improve more than epsilon, the partition process is stopped. 

        :returns: a tuple where the first element is the set of labels assigned to the left
        partition and the second element is the set of labels assigned to the right one.
        '''
        # Initialize a mean vector for the left and right partitions by uniformly sampling from all
        # the label vectors without replacement.
        mean_label_left = random.choice(list(labels_to_vectors_dict.keys()))
        mean_label_right = mean_label_left
        while mean_label_right == mean_label_left:
            mean_label_right = random.choice(list(labels_to_vectors_dict.keys()))
        mean_vector_left = labels_to_vectors_dict[mean_label_left]
        mean_vector_right = labels_to_vectors_dict[mean_label_right]

        # Initialize the similarity scores for stopping the partition process.
        old_similarity = -100
        new_similarity = 0

        # Iterate over this loop which adjusts the label assignments to the left and right
        # partitions until there is no significant improvement in assignments between consecutive
        # iterations.
        while new_similarity - old_similarity >= epsilon:
            # Get the similarities of the labels to the left and right partitions.
            node_labels_list = list(node_labels)
            (similarities_left, similarities_right, similarities) = self._get_similarities(
                mean_vector_left, mean_vector_right, node_labels_list, labels_to_vectors_dict)

            # Initialize the sets of labels that will be assigned to the left and right partitions.
            # Also, update the similarity scores of the last and this iteration.
            labelset_left = set()
            labelset_right = set()
            old_similarity = new_similarity
            new_similarity = 0

            # Sort the label indices in descending order of their similarity scores.
            sorted_indices = np.argsort(np.array(similarities) * -1)

            # Iterate over the labels to partition. The labels that have a low ranking will be
            # assigned to the left partition. The labels with a high ranking will be assigned to the
            # right partition. The label exactly in the middle will be assigned to a partition
            # based on the sign of the similarity-value of the label to the left and right clusters.
            for rank in range(len(sorted_indices)):
                label_index = sorted_indices[rank]
                label = node_labels_list[label_index]
                if rank < math.ceil(len(node_labels) / 2):
                    labelset_left.add(label)
                    new_similarity += similarities_left[label_index]
                elif rank > math.ceil(len(node_labels) / 2):
                    labelset_right.add(label)
                    new_similarity += similarities_right[label_index]
                else:
                    if similarities[rank] >= 0:
                        labelset_left.add(label)
                        new_similarity += similarities_left[label_index]
                    else:
                        labelset_right.add(label)
                        new_similarity += similarities_right[label_index]
            new_similarity /= len(sorted_indices)

            # Update the mean vectors of the left and right partitions based on the labels that were
            # assigned to them in this iteration.
            mean_vector_left = self._update_mean_vector(labelset_left, labels_to_vectors_dict)
            mean_vector_right = self._update_mean_vector(labelset_right, labels_to_vectors_dict)

        # Return sets of labels for the left and right partitions.
        return (labelset_left, labelset_right)
    

    def _get_similarities(self, mean_left, mean_right, node_labels, labels_to_vectors_dict):
        '''
        Compute the 'similarities' between a set of label vectors and the mean vectors of left and
        right partitions/clusters.

        :param mean_vector_left: the mean vector of the left partition.

        :param mean_vector_right: the mean vector of the right partition.

        :param node_labels: list of labels (strings) for which the similarities to the left and
        right partitions will be computed.

        :param labels_to_vectors_dict: a dictionary containing labels (strings) as its keys and
        their numerical vector representations as its values.

        :returns: a tuple where its first element is a list with the similarities of each label
        vector to the left partition. The second element is a list with the similarities of each
        label vector to the right partition. The third element is a list with the difference in
        similarities between the left and right partitions for each label vector.
        '''
        similarities_left = []
        similarities_right = []
        similarities = []
        for i in range(len(node_labels)):
            label = node_labels[i]
            label_vector = labels_to_vectors_dict[label].transpose()
            similarities_left.append(mean_left.dot(label_vector).toarray()[0][0])
            similarities_right.append(mean_right.dot(label_vector).toarray()[0][0])
            similarities.append(similarities_left[i] - similarities_right[i])
        return (similarities_left, similarities_right, similarities)


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
        if euclidean_norm != 0:
            updated_mean = updated_mean / euclidean_norm
        return updated_mean
