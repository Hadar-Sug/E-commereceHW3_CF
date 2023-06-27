import time
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error
import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import itertools
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsqr, svds
from sklearn.model_selection import KFold
import logging


def get_sse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    actual = np.array(actual).reshape(pred.shape)
    return mean_squared_error(pred, actual) * len(actual)


def get_rmse(pred, actual, scaler):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    actual = np.array(actual).reshape(pred.shape)
    if scaler is not None:
        pred = scaler.inverse_transform(pred.reshape(-1, 1))
        actual = scaler.inverse_transform(actual.reshape(-1, 1))
    return mean_squared_error(pred, actual) ** 0.5


def get_score(true, pred, arr_type='np'):
    score_matrix = (true - pred)
    if arr_type == 'sp':
        score_matrix = score_matrix.toarray()
    return la.norm(score_matrix) ** 2


# alternating least squares step
def als_step(latent_vectors, fixed_vecs, ratings, type='user'):
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
            latent_vectors[u, :] = la.lstsq(mat, clean_r, rcond=None)[0]
    elif type == 'item':
        # calculating Q, P is fixed
        for i in range(latent_vectors.shape[0]):
            # remove rows from
            r = ratings[:, i].flatten()
            rows_to_remove = np.where(r == 0)[0]
            clean_r = np.delete(r, rows_to_remove)
            mat = np.delete(fixed_vecs, rows_to_remove, axis=0)

            latent_vectors[i, :] = la.lstsq(mat, clean_r, rcond=None)[0]
    return latent_vectors


def create_mat(users_hash, songs_hash, df, shape):
    # df.loc[:, 'user_id'] = df['user_id'].map(users_hash).copy()
    # df.loc[:, 'song_id'] = df['song_id'].map(songs_hash).copy()
    values = df['weight'].values
    rows = df['user_id'].map(users_hash).values
    cols = df['song_id'].map(songs_hash).values
    matrix = coo_matrix((values, (rows, cols)), shape=shape)
    return matrix.tocsr()


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
        self.predictions = None
        self.train_sse = None
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
        lambda_mat = _lambda * np.eye(self.n_factors)
        zeros = np.zeros(shape=self.n_factors)
        if type == 'user':
            # calculating P, Q is fixed
            for u in range(latent_vectors.shape[0]):  # iterate over P rows and R Rows
                r = ratings[u, :].flatten()
                rows_to_remove = np.where(r == 0)[0]
                clean_r = np.delete(r, rows_to_remove)
                mat = np.delete(fixed_vecs, rows_to_remove, axis=0)
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
            self.predictions = np.maximum(predictions, 0)
        elif self.learning == 'sgd':
            global_bias_mat = np.full((self.n_users, self.n_items), self.global_bias)
            cross_product_mat = np.matmul(self.user_vecs, self.item_vecs.T)
            user_biases = np.repeat(self.user_bias, self.n_items).reshape(
                (self.n_users, self.n_items))  # duplicate to rows
            item_biases = np.tile(self.item_bias, self.n_users).reshape(
                (self.n_users, self.n_items))  # duplicate to cols
            predictions = global_bias_mat + user_biases + item_biases + cross_product_mat
            self.predictions = np.maximum(predictions, 0)

    def calculate_learning_curve(self, iter_array, test=None, learning_rate=0.1, **kwargs):
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
        # logger = kwargs['logger']
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                # logger.info(f"iteration: {n_iter}")
                print(f"iteration: {n_iter}")
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_rmse(predictions, self.ratings, self.scaler)]
            self.test_mse += [get_rmse(predictions, test, self.scaler)]
            if self._v:
                # logger.info(f"Train rmse: {str(self.train_mse[-1])}")
                # logger.info(f"Test rmse: {str(self.test_mse[-1])}")
                print(f"Train rmse: {str(self.train_mse[-1])}")
                print(f"Test rmse: {str(self.test_mse[-1])}")
            iter_diff = n_iter

    def fetch_rmse(self, test, scaler, num_iter=200):
        self.train(num_iter, learning_rate=self.learning_rate)
        predictions = self.predict_all()
        self.train_mse = get_rmse(predictions, self.ratings, scaler)
        self.test_mse = get_rmse(predictions, test, scaler)
        return self.train_mse, self.test_mse

    def fetch_sse(self, num_iter=20):
        self.train(num_iter)
        self.train_sse = get_sse(self.predictions, self.ratings)
        return self.train_sse, self.predictions


class CF:
    def __init__(self,
                 train_df,
                 test_df):
        self.train_df = train_df
        self.test_df = test_df
        train_users = train_df.iloc[:, 0].unique().tolist()
        train_songs = train_df.iloc[:, 1].unique().tolist()
        test_users = test_df.iloc[:, 0].unique().tolist()
        test_songs = test_df.iloc[:, 1].unique().tolist()
        all_users = list(set(train_users + test_users))
        all_songs = list(set(train_songs + test_songs))
        self.all_users = all_users
        self.all_songs = all_songs
        self.num_users = len(all_users)
        self.num_songs = len(all_songs)
        self.avg = train_df.iloc[:, 2].mean()
        self.users_hash = {}
        self.songs_hash_ls = {}
        self.songs_hash = {}

        for i, user in enumerate(self.all_users):
            self.users_hash[user] = i

        for j, song in enumerate(all_songs):
            self.songs_hash_ls[song] = j + self.num_users

        for j, song in enumerate(all_songs):
            self.songs_hash[song] = j

        values = train_df['weight'].values
        self.true_rows = train_df['user_id'].map(self.users_hash).values
        self.true_cols = train_df['song_id'].map(self.songs_hash).values
        self.true_mat = coo_matrix((values, (self.true_rows, self.true_cols)),
                                   shape=(self.num_users, self.num_songs)).tocsr()
        self.mat_part2 = self.true_mat.toarray()
        self.test_users_mapped = test_df['user_id'].map(self.users_hash).values
        self.test_songs_mapped = test_df['song_id'].map(self.songs_hash).values
        self.test_indxs_mapped = list((zip(self.test_users_mapped, self.test_songs_mapped)))

    def run_part_1(self):
        train_users = self.train_df['user_id'].map(self.users_hash).values
        train_songs = self.train_df['song_id'].map(self.songs_hash_ls).values
        merged_train = np.column_stack((train_users, train_songs)).reshape(-1)
        A_cols = merged_train
        rows1, rows2 = np.arange(int(0.5 * len(A_cols))), np.arange(int(0.5 * len(A_cols)))
        A_rows = np.column_stack((rows1, rows2)).reshape(-1)
        values = np.ones(len(A_rows))
        A = coo_matrix((values, (A_rows, A_cols)),
                       shape=(int(0.5 * len(A_cols)), self.num_users + self.num_songs)).tocsr()
        c = self.train_df['weight'].values - self.avg
        final_b = lsqr(A=A, b=c)
        final_b = final_b[0]
        b_u_hat = final_b[:self.num_users]
        b_i_hat = final_b[self.num_users:]
        predictions_part1 = self.true_mat.tolil(copy=True)
        for tup in zip(self.true_rows, self.true_cols):
            user, song = tup[0], tup[1]
            predictions_part1[tup] = max(self.avg + b_u_hat[user] + b_i_hat[song], 0)
        predictions_part1 = predictions_part1.tocsr()
        score_part1 = get_score(self.true_mat, predictions_part1, arr_type='sp')
        print(f'SSE in Part 1 on train set: {score_part1}')
        test_pred_1 = []
        for idx in self.test_indxs_mapped:
            user, song = idx[0], idx[1]
            test_pred_1.append(max(self.avg + b_u_hat[user] + b_i_hat[song], 0))
        self.test_df['predictions'] = test_pred_1
        self.test_df.to_csv('206567067_318880754_task1.csv', index=False)
        self.test_df = self.test_df.drop('predictions', axis=1)

    def als_predict_vecs(self, k=20, eps=300000):
        np.random.seed(0)
        new_songs = np.zeros((self.num_songs, k))
        new_users = np.random.randn(self.num_users, k)
        scores_array = list()
        start = True
        i = 1
        while start or (scores_array[-2] - scores_array[-1] >= eps):
            new_songs = als_step(new_songs,
                                 new_users,
                                 self.mat_part2,
                                 type='item')
            # new_songs = np.maximum(new_songs, 0)
            new_users = als_step(new_users,
                                 new_songs,
                                 self.mat_part2,
                                 type='user')
            meantime_pred = self.predict(new_users, new_songs)
            meantime_pred = np.maximum(meantime_pred, 0)
            scores_array.append(get_score(self.mat_part2, meantime_pred))
            i += 1
            if i == 3:
                start = False
        return new_users, new_songs, scores_array

    def predict(self, users, songs):
        predictions = np.matmul(users, songs.T)
        train_indices = list(zip(self.true_rows, self.true_cols))
        mask = np.zeros(predictions.shape, dtype=bool)
        for entry in train_indices:
            mask[entry] = True
        predictions[~mask] = 0
        return predictions

    def run_part_2(self):
        user_vecs, song_vecs, scores_arr = self.als_predict_vecs()
        score_part2 = scores_arr[-1]
        print(f'SSE in Part 2 on train set: {score_part2}')
        test_pred_2 = []
        for idx in self.test_indxs_mapped:
            user, song = idx[0], idx[1]
            user_vec, song_vec = user_vecs[user], song_vecs[song]
            pred = user_vec.dot(song_vec)
            test_pred_2.append(max(pred, 0))
        self.test_df['predictions'] = test_pred_2
        self.test_df.to_csv('206567067_318880754_task2.csv', index=False)
        self.test_df = self.test_df.drop('predictions', axis=1)

    def run_part_3(self):
        # U, S, V = svds(self.true_mat.astype(float), k=20)
        k = 20
        U, S, V = np.linalg.svd(self.true_mat.toarray(), full_matrices=False)
        # Reconstruct the matrix using k singular values/vectors
        U_reduced = U[:, :k]
        S_reduced = np.diag(S[:k])
        Vt_reduced = V[:k, :]
        reduced_R = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))
        # Test
        test_pred_3 = []
        for idx in self.test_indxs_mapped:
            user, song = idx[0], idx[1]
            test_pred_3.append(max(reduced_R[user, song], 0))
        self.test_df['predictions'] = test_pred_3
        self.test_df.to_csv('206567067_318880754_task3.csv', index=False)
        self.test_df = self.test_df.drop('predictions', axis=1)
        # Train
        predictions = reduced_R
        train_indices = list(zip(self.true_rows, self.true_cols))
        mask = np.zeros(reduced_R.shape, dtype=bool)
        for entry in train_indices:
            mask[entry] = True
        predictions[~mask] = 0
        score_part3 = get_score(self.true_mat.toarray(), predictions)
        print(f'SSE in Part 3 on train set: {score_part3}')

    def run_part_4(self, _lambda, k=35, num_iters=20):
        train_mat = create_mat(self.users_hash, self.songs_hash, self.train_df,
                               (self.num_users, self.num_songs)).toarray()
        r_avg = self.train_df.iloc[:, 2].mean()
        regularized_ALS = MF.ExplicitMF(train_mat, global_bias=r_avg, scaler=None, n_factors=k, learning='als',
                                        user_fact_reg=_lambda, item_fact_reg=_lambda)
        score_part4, predictions = regularized_ALS.fetch_sse(num_iter=num_iters)
        print(f'SSE in Part 4 on train set: {score_part4}')
        test_pred_4 = []
        for idx in self.test_indxs_mapped:
            user, song = idx[0], idx[1]
            test_pred_4.append(predictions[user, song])
        self.test_df['predictions'] = test_pred_4
        self.test_df.to_csv('206567067_318880754_task4.csv', index=False)
        self.test_df = self.test_df.drop('predictions', axis=1)

    def run_part_4_cross_validation(self, k, _lambda, n_iters, splits=5, **kwargs):
        # k: Number of latent factors
        # _lambda: Regularization parameter
        # n_iters: Number of iterations
        cross_validation = KFold(n_splits=splits)
        scores = []
        logger = kwargs['logger']
        for fold, (train_idx, test_idx) in enumerate(cross_validation.split(self.train_df)):
            train_df, test_df = self.train_df.iloc[train_idx], self.train_df.iloc[test_idx]
            train_mat = create_mat(self.users_hash, self.songs_hash, train_df,
                                   (self.num_users, self.num_songs)).toarray()
            test_mat = create_mat(self.users_hash, self.songs_hash, test_df, (self.num_users, self.num_songs)).toarray()
            r_avg = train_df.iloc[:, 2].mean()
            model = MF.ExplicitMF(train_mat, global_bias=r_avg, scaler=None, n_factors=k, learning='als'
                                  , user_fact_reg=_lambda, item_fact_reg=_lambda)
            score = model.fetch_rmse(test_mat, scaler=None, num_iter=n_iters)[1]
            scores.append(score)
            logger.info(f'fold {fold}/{splits} with (k:{k}, lambda:{_lambda}): {score:.3f}')
        logger.info(
            f'all {splits} folds avg, n_iters:{n_iters} with (k:{k}, lambda:{_lambda}): {np.average(scores):.3f}')

    def run_part_4_hyperparams(self, lambdas, Ks, check_points):
        # lambdas: List of regularization parameters to test
        # Ks: List of numbers of latent factors to test
        # check_points: List of iteration checkpoints for calculating the learning curve
        logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler('logfile.txt')
        train_df, test_df = train_test_split(self.train_df, test_size=0.2)
        r_avg = train_df.iloc[:, 2].mean()
        train_mat = create_mat(self.users_hash, self.songs_hash, train_df,
                               (self.num_users, self.num_songs)).toarray()
        test_mat = create_mat(self.users_hash, self.songs_hash, test_df, (self.num_users, self.num_songs)).toarray()
        hyperparams_dict = {(reg, k): None for reg in lambdas for k in Ks}
        for i, key in enumerate(hyperparams_dict.keys()):
            logger.info(f"Starting test {i + 1} out of {len(hyperparams_dict.keys())}")
            logger.info(f"Hyperparams: {key}")
            # print(f"starting test {i + 1} out of {len(hyperparams_dict.keys())}")
            # print(f"hyperparams: {key}")
            reg = key[0]
            k = key[1]
            model = MF.ExplicitMF(ratings=train_mat, global_bias=r_avg, scaler=None, n_factors=k, learning='als'
                                  , item_bias_reg=reg, user_bias_reg=reg)
            start_time = time.time()
            model.calculate_learning_curve(iter_array=check_points, test=test_mat, logger=logger)

    def run_part_4_hyperparams_cross_validation(self, lambdas, Ks, check_points):
        # lambdas: List of regularization parameters to test
        # Ks: List of numbers of latent factors to test
        # check_points: List of iteration checkpoints for calculating the learning curve
        # train_df, test_df = train_test_split(self.test_df, test_size=0.2)
        hyperparams_dict = {(reg, k): None for reg in lambdas for k in Ks}
        for i, key in enumerate(hyperparams_dict.keys()):
            print(f"starting test {i + 1} out of {len(hyperparams_dict.keys())}")
            print(f"hyperparams: {key}")
            train_scores = []
            test_scores = []
            cross_validation = KFold(n_splits=5)
            for fold, (train_idx, test_idx) in enumerate(cross_validation.split(self.train_df)):
                print(f'starting {fold + 1}/5 fold')
                train_df, test_df = self.train_df.iloc[train_idx], self.train_df.iloc[test_idx]
                r_avg = train_df.iloc[:, 2].mean()
                train_mat = create_mat(self.users_hash, self.songs_hash, train_df,
                                       (self.num_users, self.num_songs)).toarray()
                test_mat = create_mat(self.users_hash, self.songs_hash, test_df,
                                      (self.num_users, self.num_songs)).toarray()
                model = MF.ExplicitMF(ratings=train_mat, global_bias=r_avg, scaler=None, n_factors=key[1],
                                      learning='als'
                                      , item_fact_reg=key[0], user_fact_reg=key[0])
                model.calculate_learning_curve(iter_array=check_points, test=test_mat)
                train_scores.append(model.train_mse)
                test_scores.append(model.test_mse)
            train_array = np.array(train_scores)
            averages_train = np.mean(train_array, axis=0).tolist()
            test_array = np.array(test_scores)
            averages_test = np.mean(test_array, axis=0).tolist()
            df_scores = pd.DataFrame(data={"train": averages_train, "test": averages_test}, index=check_points)
            print(df_scores.T)
            plt.plot(check_points, averages_train, label='train')
            plt.plot(check_points, averages_test, label='test')
            plt.title(f'Train&Test scores with hyperparams: {key}')
            plt.xlabel('ALS iterations')
            plt.ylabel('RMSE')
            plt.legend()
            plt.show()


def main():
    part1_2_3 = CF(pd.read_csv('user_song.csv'), pd.read_csv('test.csv'))
    part1_2_3.run_part_1()
    part1_2_3.run_part_2()
    part1_2_3.run_part_3()


if __name__ == "__main__":
    main()
