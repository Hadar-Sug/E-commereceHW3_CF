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
    df = pd.read_csv('user_songs_edited.csv')
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['weight'] = scaler.fit_transform(df[['weight']])
    train_df, test_df = train_test_split(df, test_size=0.2)
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
    return df, train_mat, test_mat


def create_mat(users_hash, songs_hash, df, shape):
    # df.loc[:, 'user_id'] = df['user_id'].map(users_hash).copy()
    # df.loc[:, 'song_id'] = df['song_id'].map(songs_hash).copy()
    values = df['weight'].values
    rows = df['user_id'].map(users_hash).values
    cols = df['song_id'].map(songs_hash).values
    matrix = coo_matrix((values, (rows, cols)), shape=shape)
    return matrix.tocsr()


def main():
    df, train, test = init()
    r_avg = df.iloc[:, 2].mean()
    regulerization = [0.01, 0.1, 0.2, 0.33]
    learning_rates = [1e-3, 1e-2, 0.1]
    K_list = [20, 50, 100]
    # regulerization = [0.01]
    # learning_rates = [1e-2]
    # K_list = [50]
    hyperparams_dict = {(reg, rate, k): None for reg in regulerization[::-1] for rate in learning_rates for k in K_list}
    for i,key in enumerate(hyperparams_dict.keys()):
        print(f"starting test {i} out of {len(hyperparams_dict.keys())}")
        reg = key[0]
        rate = key[1]
        k = key[2]
        SGD = ExplicitMF(ratings=train, global_bias=r_avg, n_factors=k, learning='sgd', user_fact_reg=reg, \
                         user_bias_reg=reg,
                         item_fact_reg=reg, item_bias_reg=reg, learning_rate=rate)
        hyperparams_dict[key] = SGD.fetch_mse(test)
        print(key, hyperparams_dict[key])
    best_train = min(hyperparams_dict, key=lambda k: hyperparams_dict[k][0])
    best_test = min(hyperparams_dict, key=lambda k: hyperparams_dict[k][1])
    print(f"best train params: {best_train}\n best test params: {best_test}")
    print(hyperparams_dict)


if __name__ == "__main__":
    main()
