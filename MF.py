import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from numpy.linalg import solve
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize


class ExplicitMF:
    def __init__(self,
                 ratings,
                 global_bias,
                 scaler,
                 n_factors=40,
                 learning='sgd',
                 learning_rate=0.0,
                 item_fact_reg=0.0,
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=True):

        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model
        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.

        item_fact_reg : (float)
            Regularization term for item latent factors

        user_fact_reg : (float)
            Regularization term for user latent factors

        item_bias_reg : (float)
            Regularization term for item biases

        user_bias_reg : (float)
            Regularization term for user biases

        verbose : (bool)
            Whether or not to printout training progress
        """
        self.scaler = scaler
        self.test_mse = None
        self.train_mse = None
        self.training_indices = None
        self.global_bias = global_bias
        self.item_bias = None
        self.user_bias = None
        self.learning_rate = learning_rate
        self.item_vecs = None
        self.user_vecs = None
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning = learning
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose
        random.seed(0)
        np.random.seed(0)

    # Define the objective function

    # TODO replace with working version
    def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # calculating P, Q is fixed
            for u in range(latent_vectors.shape[0]):  # iterate over P rows and R Rows
                r = ratings[u, :].flatten()
                rows_to_remove = np.where(r == 0)[0]
                clean_r = np.delete(r, rows_to_remove)
                mat = np.delete(fixed_vecs, rows_to_remove, axis=0)
                lambda_mat = _lambda * np.eye(self.n_factors)
                zeros = np.zeros(shape=self.n_factors)
                mat = np.vstack((mat, lambda_mat))
                b = np.hstack((clean_r, zeros)).flatten()
                latent_vectors[u, :] = la.lstsq(mat, b, rcond=None)[0]
        elif type == 'item':
            # calculating Q, P is fixed
            for i in range(latent_vectors.shape[0]):
                # remove rows from
                r = ratings[:, i].flatten()
                rows_to_remove = np.where(r == 0)[0]
                clean_r = np.delete(r, rows_to_remove)
                mat = np.delete(fixed_vecs, rows_to_remove, axis=0)
                lambda_mat = _lambda * np.eye(self.n_factors)
                zeros = np.zeros(shape=self.n_factors)
                mat = np.vstack((mat, lambda_mat))
                b = np.hstack((clean_r, zeros)).flatten()
                latent_vectors[i, :] = la.lstsq(mat, b, rcond=None)[0]
        return latent_vectors

    def train(self, n_iter=10, learning_rate=0.1):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        np.random.seed(0)
        self.user_vecs = np.random.randn(self.n_users, self.n_factors)
        self.item_vecs = np.zeros((self.n_items, self.n_factors))

        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        np.random.seed(0)
        for i in range(n_iter):
            if self.learning == 'als':
                self.item_vecs = self.als_step(self.item_vecs,
                                               self.user_vecs,
                                               self.ratings,
                                               self.item_fact_reg,
                                               type='item')
                self.user_vecs = self.als_step(self.user_vecs,
                                               self.item_vecs,
                                               self.ratings,
                                               self.user_fact_reg,
                                               type='user')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()

    def sgd(self):
        for m, idx in enumerate(self.training_indices):
            # if m <= 6:
            #     print(idx)
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u, i] - prediction)  # error
            # Update biases

            self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])

            self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])
            # Update latent factors
            self.user_vecs[u, :] += self.learning_rate * \
                                    (e * self.item_vecs[i, :] - self.user_bias_reg * self.user_vecs[u, :])
            self.item_vecs[i, :] += self.learning_rate * (
                    e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i, :])

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return max(0, prediction)

    def predict_all(self):
        """ Predict ratings for every user and item."""
        if self.learning == 'als':
            predictions = np.matmul(self.user_vecs, self.item_vecs.T)
            return np.maximum(predictions, 0)
        elif self.learning == 'sgd':
            global_bias_mat = np.full((self.n_users, self.n_items), self.global_bias)
            cross_product_mat = np.matmul(self.user_vecs, self.item_vecs.T)
            user_biases = np.repeat(self.user_bias, self.n_items).reshape(
                (self.n_users, self.n_items))  # duplicate to rows
            item_biases = np.tile(self.item_bias, self.n_users).reshape(
                (self.n_users, self.n_items))  # duplicate to cols
            predictions = global_bias_mat + user_biases + item_biases + cross_product_mat
            return np.maximum(predictions, 0)

    def calculate_learning_curve(self, iter_array, test=None, learning_rate=0.1):
        """
        Keep track of MSE as a function of training iterations.

        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).

        The function creates two new class attributes:

        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print(f"iteration: {n_iter}")
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_rmse(predictions, self.ratings, self.scaler)]
            self.test_mse += [get_rmse(predictions, test, self.scaler)]
            if self._v:
                print(f"Train rmse: {str(self.train_mse[-1])}")
                print(f"Test rmse: {str(self.test_mse[-1])}")
            iter_diff = n_iter

    def fetch_rmse(self, test, scaler, num_iter=200):
        self.train(num_iter, learning_rate=self.learning_rate)
        predictions = self.predict_all()
        self.train_mse = get_rmse(predictions, self.ratings, scaler)
        self.test_mse = get_rmse(predictions, test, scaler)
        return self.train_mse, self.test_mse


def get_rmse(pred, actual, scaler):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    actual = np.array(actual).reshape(pred.shape)
    # pred = scaler.inverse_transform(pred.reshape(-1, 1))
    # actual = scaler.inverse_transform(actual.reshape(-1, 1))
    return mean_squared_error(pred, actual) ** 0.5
