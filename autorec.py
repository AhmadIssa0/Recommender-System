

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, TimeDistributed, InputLayer, Bidirectional, Softmax, BatchNormalization, Lambda, LayerNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import numpy as np
from movielens_recommender import MovieLens

class AutoRec:
    """ Tensorflow implementation of the auto-encoder recommender system described in 
        "AutoRec: Autoencoders Meet Collaborative Filtering"
        by Suvash Sedhain, Aditya Krishna Menon, Scott Sanner and Lexing Xie.
    """

    def __init__(self, m, k, max_rating=5., alpha=0.3, activation1='relu', activation2='sigmoid'):
        """
        :param m: num inputs to autoencoder, in item-based recommender system this is number of users.
        :param k: num of dimension in encoding.
        :param alpha: regularization strength.
        :param max_rating: ratings are assumed to be in [1, max_rating], used for scaling purposes.
        :param activation1: first (lowest) layer activation function.
        :param activation2: second (highest) layer activation function.
        """
        self.m = m
        self.k = k 
        self.max_rating = max_rating 
        self.alpha = alpha
        self.activation1 = activation1
        self.activation2 = activation2

    def copy(self):
        autorec = AutoRec(m=self.m, k=self.k, max_rating=self.max_rating, alpha=self.alpha,
                          activation1=self.activation1, activation2=self.activation2)
        autorec.model = keras.models.clone_model(self.model)
        autorec.model.set_weights(self.model.get_weights())
        return autorec
        
    def create_model(self):
        self.model = Sequential([
            InputLayer(input_shape=(self.m,)),
            #Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
            Dense(self.k, activation=self.activation1, kernel_regularizer=tf.keras.regularizers.L2(self.alpha)),
            Dense(self.m, activation=self.activation2, kernel_regularizer=tf.keras.regularizers.L2(self.alpha)),
            Lambda(lambda x: x * (self.max_rating - 1.0) + 1.0) # scale into range [1, max_rating]
        ])

    def training_step(self, batch, batch_mask, optimizer=keras.optimizers.Adam(lr=1e-4)):
        batch_size = tf.constant(batch.shape[0], dtype='float32')
        #batch = batch / self.max_rating
        batch = tf.constant(batch)
        with tf.GradientTape() as tape:
            preds = self.model(batch, training=True)
            squared_residuals = tf.square((preds - batch) * batch_mask)
            pred_loss = tf.reduce_sum(squared_residuals) / batch_size
            reg_loss = tf.reduce_sum(self.model.losses)
            total_loss = pred_loss + reg_loss
            
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return (total_loss.numpy(), pred_loss.numpy(), reg_loss.numpy())

    def rmse(self, mat, training_mask, testing_mask=None):
        """
        Returns RMSE of model on entries indicated by `testing_mask`.
        :param training_mask: Indicator matrix for entries which are visible to and input into the model.
        """
        resid = (self.model.predict(mat * training_mask) - mat) * testing_mask
        sq_resid = tf.square(resid)
        mse = tf.reduce_sum(sq_resid) / tf.reduce_sum(testing_mask)
        rmse = np.sqrt(mse.numpy())
        return rmse

    def fit(self, mat, training_mask, max_epochs=2000, alpha=0.3, optimizer=keras.optimizers.Adam(lr=1e-3),
            verbose=True, validation_mask=None, early_stop=True, patience=20, batch_size=100):
        """
        Trains the model.
        :param mat: Numpy matrix of ratings. Must have `m` columns.
        :param training_mask: Indicator matrix indicating entries of `mat` that will be used for training.
        :param validation_mask: Indicator matrix indicating entries to be used as validation data.
        """
        mat = mat.astype(np.float32)
        train_mask = mat.astype(np.float32)
        n = mat.shape[0] # num items      
        epoch = 0
        early_stopped = False
        best_validation_rmse = np.inf
        best_rmse_epoch = 0
        best_model = self.copy()
        
        while epoch < max_epochs and (not early_stop or not early_stopped):
            batches = tf.data.Dataset.range(n).batch(batch_size).as_numpy_iterator()
            epoch_loss = 0.0
            for batch in batches:
                xs = mat[batch, :]
                xs_mask = training_mask[batch, :]
                #print('xs shape', xs.shape)
                losses = self.training_step(xs * xs_mask, xs_mask, optimizer=optimizer)
                epoch_loss += losses[1]
            
            if validation_mask is not None:
                validation_rmse = self.rmse(mat, training_mask, validation_mask)
                if validation_rmse >= best_validation_rmse: # didn't improve
                    if epoch - best_rmse_epoch > patience:
                        early_stopped = True
                else:
                    best_validation_rmse = validation_rmse
                    best_rmse_epoch = epoch
                    best_model = self.copy()
                validation_rmse = f'{validation_rmse:.5f}'
            else:
                validation_rmse = None
            if verbose:
                print(f'Epoch {epoch} of {max_epochs}. Epoch loss: {epoch_loss:.5f}, '
                      f'validation rmse: {validation_rmse}, train rmse: {self.rmse(mat, training_mask, training_mask):.5f}')
            epoch += 1
        if verbose:
            print(f'Best validation rmse: {best_validation_rmse:.5f}')
        return best_model, best_validation_rmse
                    

def train_movielens(k=50, epochs=10, batch_size=50, alpha=0.3, optimizer=keras.optimizers.Adam(lr=1e-3)):
    ml = MovieLens()
    mat = ml.mat.transpose().astype('float32') # shape [items, users]
    train_mask, test_mask = ml.train_test_split(mask=ml.mat_ind.transpose())
    train_mask = train_mask.astype(np.float32)
    test_mask = test_mask.astype(np.float32)
    #mat_ind = ml.mat_ind.transpose().astype('float32')
    m = mat.shape[1] # num users
    autorec = AutoRec(m=m, k=k, max_rating=5.0, alpha=alpha)
    autorec.create_model()
    n = mat.shape[0] # num movies

    
    def rmse(mask=None):
        if mask is None:
            mask = test_mask
        resid = (autorec.model.predict(mat * train_mask) - mat) * mask
        sq_resid = tf.square(resid)
        mse = tf.reduce_sum(sq_resid) / tf.reduce_sum(mask)
        rmse = np.sqrt(mse.numpy())
        #print(f'RMSE on test set: {rmse}')
        return rmse
        
    for epoch in range(epochs):
        batches = tf.data.Dataset.range(n).batch(batch_size).as_numpy_iterator()
        epoch_loss = 0.0
        for batch in batches:
            xs = mat[batch, :]
            xs_mask = train_mask[batch, :]
            # print('xs shape', xs.shape)
            losses = autorec.training_step(xs * xs_mask, xs_mask, optimizer=optimizer)
            epoch_loss += losses[1]
        print(f'Epoch {epoch} of {epochs}. Loss per movie: {epoch_loss / n}, test rmse: {rmse():.5f}, train rmse: {rmse(train_mask):.5f}')

    return autorec
        
def test_example():
    autorec = AutoRec(m=4, k=2, max_rating=1.0)
    autorec.create_model()

    ratings = np.array(
    [[0.1, 0.9, 1.0],
     [0.3, 0.8, 0.8],
     [0.8, 0.1, 0.2],
     [0.7, 0.1, 0.1]], dtype='float32').transpose()

    ratings_mask = np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 0],
     [1, 0, 1]], dtype='float32').transpose()

    for i in range(1000):
        autorec.training_step(batch=ratings, batch_mask=ratings_mask)
        
    return autorec

