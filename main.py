from scipy.sparse import coo_matrix

import MF
from MF import ExplicitMF
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
        test_df = self.test_df.drop('predictions', axis=1)

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
        U, S, V = svds(self.true_mat.astype(float), k=20)
        # Reconstruct the matrix using k singular values/vectors
        reduced_R = U.dot(diags(S).dot(V))
        predictions = reduced_R
        train_indices = list(zip(self.true_rows, self.true_cols))
        mask = np.zeros(reduced_R.shape, dtype=bool)
        for entry in train_indices:
            mask[entry] = True
        predictions[~mask] = 0
        score_part3 = get_score(self.true_mat.toarray(), predictions)
        print(f'SSE in Part 3 on train set: {score_part3}')
        test_pred_3 = []
        for idx in self.test_indxs_mapped:
            user, song = idx[0], idx[1]
            test_pred_3.append(max(reduced_R[user, song], 0))
        self.test_df['predictions'] = test_pred_3
        self.test_df.to_csv('206567067_318880754_task3.csv', index=False)
        self.test_df = self.test_df.drop('predictions', axis=1)

    def run_part_4_cross_validation(self, k, _lambda, n_iters):
        cross_validation = KFold(n_splits=5)
        for train_idx, test_idx in cross_validation.split(self.train_df):
            train_df, test_df = self.train_df.iloc[train_idx], self.train_df.iloc[test_idx]
            train_mat = create_mat(self.users_hash, self.songs_hash, train_df,
                                   (self.num_users, self.num_songs)).toarray()
            test_mat = create_mat(self.users_hash, self.songs_hash, test_df, (self.num_users, self.num_songs)).toarray()
            r_avg = train_df.iloc[:, 2].mean()
            model = MF.ExplicitMF(train_mat, global_bias=r_avg, scaler=None, n_factors=k, learning='als'
                                  , item_bias_reg=-_lambda, user_bias_reg=_lambda)
            model.calculate_learning_curve(np.arange(1, n_iters, step=5), test_mat)

    def run_part_4_hyperparams(self, lambdas, Ks, check_points):
        train_df, test_df = train_test_split(self.test_df, test_size=0.2)
        r_avg = train_df.iloc[:, 2].mean()
        train_mat = create_mat(self.users_hash, self.songs_hash, train_df,
                               (self.num_users, self.num_songs)).toarray()
        test_mat = create_mat(self.users_hash, self.songs_hash, test_df, (self.num_users, self.num_songs)).toarray()
        hyperparams_dict = {(reg, k): None for reg in lambdas for k in Ks}
        for i, key in enumerate(hyperparams_dict.keys()):
            print(f"starting test {i + 1} out of {len(hyperparams_dict.keys())}")
            print(f"hyperparams: {key}")
            reg = key[0]
            k = key[1]
            model = MF.ExplicitMF(ratings=train_mat, global_bias=r_avg, scaler=None, n_factors=k, learning='als'
                                  , item_bias_reg=reg, user_bias_reg=reg)
            model.calculate_learning_curve(iter_array=check_points, test=test_mat)


def main(q=4):
    if q == 4:
        part4 = CF(pd.read_csv('user_song.csv'), pd.read_csv('test.csv'))
        regularization = np.arange(start=0.01, stop=0.5, step=0.05)
        K_list = np.arange(start=20, stop=60, step=5)
        checkpoints = [1, 2, 5, 10, 20, 50]
        part4.run_part_4_hyperparams(lambdas=regularization, Ks=K_list, check_points=checkpoints)
    else:
        part1_2_3 = CF(pd.read_csv('user_song.csv'), pd.read_csv('test.csv'))
        part1_2_3.run_part_1()
        part1_2_3.run_part_2()
        part1_2_3.run_part_3()


if __name__ == "__main__":
    main()
