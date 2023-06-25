# this file is old code

# def init():
#     train = pd.read_csv('user_song.csv')
#     test_df = pd.read_csv('test.csv')
#     train_users = train.iloc[:, 0].unique().tolist()
#     train_songs = train.iloc[:, 1].unique().tolist()
#
#     test_users = test_df.iloc[:, 0].unique().tolist()
#     test_songs = test_df.iloc[:, 1].unique().tolist()
#
#     all_users = list(set(train_users + test_users))
#     all_songs = list(set(train_songs + test_songs))
#     # scaler = MinMaxScaler(feature_range=(0, 10))
#     # train['weight'] = scaler.fit_transform(train[['weight']])
#     train_df, test_df = train_test_split(train, test_size=0.2)
#     r_avg = train_df.iloc[:, 2].mean()
#     num_users = len(all_users)  # Number of users
#     num_songs = len(all_songs)
#     # user dictionary
#     users_hash = {}
#     for i, user in enumerate(all_users):
#         users_hash[user] = i
#     # song dictionary
#     songs_hash = {}
#     for j, song in enumerate(all_songs):
#         songs_hash[song] = j
#     train_mat = create_mat(users_hash, songs_hash, train_df, (num_users, num_songs))
#     test_mat = create_mat(users_hash, songs_hash, test_df, (num_users, num_songs))
#     return train, train_mat, test_mat, r_avg, scaler

# main():
# np.random.seed(0)
# df, train, test, r_avg, scaler = init()
# r_avg = df.iloc[:, 2].mean()
# regularization = np.arange(start=0.01, stop=0.5, step=0.05)  # best train params: (0.01, 0.03, 20)
# learning_rates = [0]  # best test params: (0.33, 0.03, 20)
# K_list = np.arange(start=30, stop=85, step=5)
# hyperparams_dict = {(reg, k): None for reg in regularization[::-1] for k in K_list}
# for i, key in enumerate(hyperparams_dict.keys()):
#     print(f"starting test {i + 1} out of {len(hyperparams_dict.keys())}")
#     print(f"hyperparams: {key}")
#     reg = key[0]
#     k = key[1]
#     SGD = ExplicitMF(ratings=train.toarray(), global_bias=r_avg, n_factors=k, learning='als', user_fact_reg=reg,
#                      user_bias_reg=reg,
#                      item_fact_reg=reg, item_bias_reg=reg, scaler=scaler)
#     SGD.calculate_learning_curve([1, 2, 5, 10, 20, 50], test.toarray())
# #     hyperparams_dict[key] = SGD.fetch_rmse(test, scaler =scaler)
# #     print(key, hyperparams_dict[key])
# # best_train = min(hyperparams_dict, key=lambda k: hyperparams_dict[k][0])
# # best_test = min(hyperparams_dict, key=lambda k: hyperparams_dict[k][1])
# # print(f"best train params: {best_train}\n best test params: {best_test}")
# # print(hyperparams_dict)