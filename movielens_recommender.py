

# run this in Anaconda.

import numpy as np
import pandas as pd
import pickle
from matrix_factorization import MatrixFactorization

class MovieLens:

    def __init__(self, folder='C:/Users/Ahmad/Documents/Jupyter notebooks/movielens/ml-100k'):
        """ :param folder: Folder path containing movielens files.
        """
        # Items contains movie information.
        self.items_cols = "movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western"
        self.items_cols = self.items_cols.split(' | ')
        self.items = pd.read_csv(folder + '/u.item', sep='|', encoding='iso-8859-1', names=self.items_cols)
        # in order of movie id
        self.movie_titles = self.items['movie title'].tolist()
        
        # Contains rating information.
        self.data_cols = "user id | item id | rating | timestamp".split(' | ')
        self.data = pd.read_csv(folder + '/u.data', sep='\t', encoding='iso-8859-1', names=self.data_cols)

        self.mat = np.zeros((943,1682)) # 943 users, 1682 movies
        self.mat_ind = np.zeros((943,1682), dtype=np.int32)
        for row in self.data[['user id', 'item id', 'rating']].to_numpy():
            user_id, item_id, rating = row[0]-1, row[1]-1, row[2]
            self.mat[user_id][item_id] = rating
            self.mat_ind[user_id][item_id] = 1

        self.u, self.v, self.mu, self.rank = None, None, None, None

    def reset_model(self):
        self.u, self.v, self.mu, self.rank = None, None, None, None

    def train_test_split(self, mask=None, test_size=0.05):
        """ Returns train test split of mask as 0/1 matrices.
            :param test_size: float which is the probability of including a sample in the test set.
            :param mask: Matrix that gets split. If None, then uses mat_ind.
        """
        if mask is None:
            mask = self.mat_ind
            
        test_mask, train_mask = np.zeros(mask.shape, dtype=np.int32), np.zeros(mask.shape, dtype=np.int32)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] == 1:
                    if np.random.random() < test_size:
                        test_mask[i,j] = 1
                    else:
                        train_mask[i,j] = 1
        return (train_mask, test_mask)

    def compute_factorization(self, rank, mat_ind=None, **optional_params):
        """ Compute the rating matrix as 
                  Ratings matrix = U * V^T + mu.
            :param mat_ind: If not None, specifies 0/1 matrix of entries which the model trains on. Used for computing test error.
        """
        # Ratings matrix = U * V^T + mu
        if mat_ind is None:
            mat_ind = self.mat_ind
        self.u, self.v, self.mu = MatrixFactorization.sgd_factorize(rank, self.mat, mat_ind, **optional_params)
        self.rank = rank

    def rmse(self, mat_ind=None):
        if mat_ind is None:
            mat_ind = self.mat_ind
        import math
        return math.sqrt(MatrixFactorization.mse(self.u, self.v, self.mat, mat_ind, self.mu))

    def pickle_factorization(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump((self.u, self.v, self.mu, self.rank), outfile)

    def unpickle_factorization(self, filename):
        with open(filename, 'rb') as infile:
            self.u, self.v, self.mu, self.rank = pickle.load(infile)

    def user_ratings(self, user_id):
        # Note that user_id starts from 1 rather than 0.
        i = user_id - 1
        ratings = self.u[i] @ self.v.transpose() + self.mu
        #ratings = [(self.movie_titles[i], ratings[i]) for i in range(ratings.shape[0])]
        #ratings.sort(key=lambda x: -x[1])
        res = pd.DataFrame()
        res['movie title'] = self.items['movie title']
        res['predicted rating'] = pd.DataFrame(ratings.transpose())
        res['rating'] = pd.DataFrame(self.mat[i].transpose())
        return res

    def movie_biases(self):
        res = pd.DataFrame()
        res['movie title'] = self.items['movie title']
        res['bias'] = pd.DataFrame(self.v[:,-1])
        return res
        
