import numpy as np
import sklearn


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (num_train,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, k=1, num_loops=0, distance_type=2):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X_test, distance_type)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X_test, distance_type)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X_test, distance_type)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X_test, distance_type=2):
        """
        Compute the distance between each test point in X_test and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 and l1 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # pass
        if distance_type == 2:

            test_sum = np.sum(np.square(X_test), axis=1)  # (num_test,)
            train_sum = np.sum(np.square(self.X_train), axis=1)  # (num_train,)
            inner_product = np.dot(X_test, self.X_train.T)  # num_test x num_train
            # print(test_sum.shape, train_sum.shape, inner_product.shape)
            dists = np.sqrt(-2 * inner_product + train_sum + np.transpose([test_sum]))

            # dists = np.sqrt(-2 * np.dot(X_test, self.X_train.T) + np.sum(np.square(self.X_train), axis=1) +
            #                 np.transpose([np.sum(np.square(X_test), axis=1)]))

        elif distance_type == 1:
            # L1 distance I don't know how to realize
            dists = np.sqrt(-2 * np.dot(X_test, self.X_train.T) + np.sum(np.square(self.X_train), axis=1) +
                            np.transpose([np.sum(np.square(X_test), axis=1)]))
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def compute_distances_one_loop(self, X_test, distance_type=2):
        """
        Compute the distance between each test point in X_test and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            # pass
            if distance_type == 2:
                dists[i] = np.sqrt(np.sum(np.square(self.X_train - X_test[i]), axis=1))
                # dists[i] = np.sum(np.square(self.X_train - X_test[i]), axis=1)
            elif distance_type == 1:
                dists[i] = np.sum(np.abs(self.X_train - X_test[i]), axis=1)

            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_two_loops(self, X_test, distance_type=2):
        """
        Compute the distance between each test point in X_test and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                # pass
                if distance_type == 2:
                    dists[i][j] = np.sqrt(np.sum(np.square(X_test[i] - self.X_train[j])))
                    # dists[i][j] = np.sum(np.square(X_test[i] - self.X_train[j]))
                elif distance_type == 1:
                    dists[i][j] = np.sum(np.abs(X_test[i] - self.X_train[j]))
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # pass
            # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引号)
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # pass
            # np.bincount返回的是0–序列最大值在这个array中出现的次数,array必须是一维且只包含非负整数
            y_pred[i] = np.argmax(np.bincount(closest_y))
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred
