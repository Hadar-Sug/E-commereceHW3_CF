from scipy.sparse import coo_matrix

from MF import ExplicitMF
import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import itertools

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsqr, svds
import numpy.linalg as la


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


class CF:
    def __init__(self,
                 train_df,
                 test_df):
        self.train_df = train_df
        self.test_df = test_df
        train_users = train_df.iloc[:, 0].unique().tolist()
        train_songs = train_df.iloc[:, 1].unique().tolist()
        train_couples = list(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))
        test_users = test_df.iloc[:, 0].unique().tolist()
        test_songs = test_df.iloc[:, 1].unique().tolist()
        test_couples = list(zip(test_df.iloc[:, 0], test_df.iloc[:, 1]))
        all_users = list(set(train_users + test_users))
        all_songs = list(set(train_songs + test_songs))
        all_couples = list(itertools.product(all_users, all_songs))
        new_couples = list(set(all_couples) - set(train_couples) - set(test_couples))
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


def init():
    train = pd.read_csv('user_song.csv')
    test_df = pd.read_csv('test.csv')
    train_users = train.iloc[:, 0].unique().tolist()
    train_songs = train.iloc[:, 1].unique().tolist()

    test_users = test_df.iloc[:, 0].unique().tolist()
    test_songs = test_df.iloc[:, 1].unique().tolist()

    all_users = list(set(train_users + test_users))
    all_songs = list(set(train_songs + test_songs))
    # test_couples = list(zip(test_df.iloc[:, 0], test_df.iloc[:, 1]))
    # train_couples = list(zip(train.iloc[:, 0], train.iloc[:, 1]))
    # all_couples = list(itertools.product(all_users, all_songs))
    # new_couples = list(set(all_couples) - set(train_couples) - set(test_couples))
    # new_couples = random.sample(new_couples, k=int(len(new_couples) * 0.01))  # taking a fraction of zeros
    # df_art = pd.DataFrame(new_couples, columns=['user_id', 'song_id'])
    # print("here")
    # df_art['weight'] = 0
    # df = pd.concat([df, df_art], ignore_index=True)
    # df = pd.read_csv('user_songs_edited.csv')
    scaler = MinMaxScaler(feature_range=(0, 10))
    train['weight'] = scaler.fit_transform(train[['weight']])
    train_df, test_df = train_test_split(train, test_size=0.2)
    r_avg = train_df.iloc[:, 2].mean()
    num_users = len(all_users)  # Number of users
    num_songs = len(all_songs)
    # create the sparse ratings matrix
    R = sp.lil_matrix((num_users, num_songs))
    # user dictionary
    users_hash = {}
    for i, user in enumerate(all_users):
        users_hash[user] = i
    # song dictionary
    songs_hash = {}
    for j, song in enumerate(all_songs):
        songs_hash[song] = j
    train_mat = create_mat(users_hash, songs_hash, train_df, (num_users, num_songs))
    test_mat = create_mat(users_hash, songs_hash, test_df, (num_users, num_songs))
    return train, train_mat, test_mat, r_avg, scaler


def create_mat(users_hash, songs_hash, df, shape):
    # df.loc[:, 'user_id'] = df['user_id'].map(users_hash).copy()
    # df.loc[:, 'song_id'] = df['song_id'].map(songs_hash).copy()
    values = df['weight'].values
    rows = df['user_id'].map(users_hash).values
    cols = df['song_id'].map(songs_hash).values
    matrix = coo_matrix((values, (rows, cols)), shape=shape)
    return matrix.tocsr()


def main(q=4):
    if q == 4:
        np.random.seed(0)
        df, train, test, r_avg, scaler = init()
        r_avg = df.iloc[:, 2].mean()
        regularization = [0.2]  # best train params: (0.01, 0.03, 20)
        learning_rates = [0.02]  # best test params: (0.33, 0.03, 20)
        K_list = [100]
        hyperparams_dict = {(reg, rate, k): None for reg in regularization[::-1] for rate in learning_rates for k in
                            K_list}
        for i, key in enumerate(hyperparams_dict.keys()):
            print(f"starting test {i + 1} out of {len(hyperparams_dict.keys())}")
            reg = key[0]
            rate = key[1]
            k = key[2]
            SGD = ExplicitMF(ratings=train, global_bias=r_avg, n_factors=k, learning='sgd', user_fact_reg=reg, \
                             user_bias_reg=reg,
                             item_fact_reg=reg, item_bias_reg=reg, learning_rate=rate, scaler=scaler)
            SGD.calculate_learning_curve([1, 2, 5, 10, 20, 50, 100], test, learning_rate=rate)
        #     hyperparams_dict[key] = SGD.fetch_rmse(test, scaler =scaler)
        #     print(key, hyperparams_dict[key])
        # best_train = min(hyperparams_dict, key=lambda k: hyperparams_dict[k][0])
        # best_test = min(hyperparams_dict, key=lambda k: hyperparams_dict[k][1])
        # print(f"best train params: {best_train}\n best test params: {best_test}")
        # print(hyperparams_dict)
    else:
        part1_2_3 = CF(pd.read_csv('user_song.csv'), pd.read_csv('test.csv'))
        part1_2_3.run_part_1()
        part1_2_3.run_part_2()
        part1_2_3.run_part_3()


if __name__ == "__main__":
    main(q=123)
