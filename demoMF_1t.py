# -*- coding: utf-8 -*- 
import pandas as pd
from MF import MF


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ratings.csv', sep='\t', names=r_cols, encoding='latin-1',header=0)
ratings = ratings_base.as_matrix()
ratings[:, :2] -= 1

from sklearn.model_selection import train_test_split

rate_train, rate_test = train_test_split(ratings, test_size=0.3, random_state=42)

print (rate_train.shape, rate_test.shape)

# item base
rs = MF(rate_train, K = 2, lam = 0.1, print_every = 2, learning_rate = 2, max_iter = 10, user_based = 0)
rs.fit()

RMSE = rs.evaluate_RMSE(rate_test)
print ('\nItem-based MF, RMSE =', RMSE)



# user base
# rs = MF(rate_train, K = 2, lam = 0.1, print_every = 10, 
#     learning_rate = 2 , max_iter = 10, user_based = 1)
# rs.fit()
# evaluate on test data
# RMSE = rs.evaluate_RMSE(rate_test)
# print ('\nUser-based MF, RMSE =', RMSE)