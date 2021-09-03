

import numpy as np
import random

class MatrixFactorization:
    """ Low-rank matrix factorization methods for collaborative filtering based recommender systems. """
    
    @staticmethod
    def mse(u, v, mat, mat_ind, rating_global_mean=0.0):
        prod = u @ v.transpose() + rating_global_mean
        m, n = mat.shape
        err = 0.0
        count = 0
        for i in range(m):
            for j in range(n):
                if mat_ind[i][j] == 1:
                    count += 1
                    err += (mat[i][j] - prod[i][j]) ** 2
        return err / count

    @staticmethod
    def squared_error(u, v, mat, mat_ind, rating_global_mean=0.0):
        prod = u @ v.transpose() + rating_global_mean
        m, n = mat.shape
        err = 0.0

        for i in range(m):
            for j in range(n):
                if mat_ind[i][j] == 1:                  
                    err += (mat[i][j] - prod[i][j]) ** 2
        return err

    @staticmethod
    def err_mat(u, v, mat, mat_ind):
        error = mat - u @ v.transpose() # note this is incorrect on undefined entries

        for i in range(error.shape[0]):
            for j in range(error.shape[1]):
                if mat_ind[i][j] == 0:
                    error[i][j] = 0
        return error

    @staticmethod
    def loss_function(u, v, mat, mat_ind, l2_constant, rating_global_mean=0.0):
        # Objective function = SE + Regularization
        mse = MatrixFactorization.squared_error(u, v, mat, mat_ind, rating_global_mean)
        # We subtract off m+n below, since u and v have constant 1 columns for the user-item biases.
        reg = l2_constant * (np.linalg.norm(u) ** 2 + np.linalg.norm(v) ** 2 - mat.shape[0] - mat.shape[1])
        return mse + reg

    @staticmethod
    def sgd_factorize(rank, mat, mat_ind, alpha=0.005, l2_constant=0.3, u_init=None, v_init=None,
                      momentum=0.9, iterations_max=500, bold_driver=True, print_iters=False):
        """ Factorise (partially defined) matrix mat as U x V^T + mu, where mat is m x n, U is m x (rank + 2), 
            V is n x (rank + 2), and mu is the global mean of mat over all defined entries.
            Incorporates data driven user-item biases: the last column of U consist of 1's, similarly
            the second-last column of V.
            Implements stochastic gradient descent with the bold driver algorithm and Nesterov optimizer.

            :param mat: The numpy matrix of entries.
            :param mat_ind: Indicator matrix mat_ind[i][j] = 1 indicates that mat[i][j] is defined.
            :param alpha: The initial stochastic gradient descent step size.
            :param l2_constant: is the L2 regularization constant.
            :returns: (U, V, mu), where mu is the global mean of mat over all defined entries.
        """
        m, n = mat.shape

        # TODO: Use SVD to get a better initial guess.
        """ rank + 2 is used since the last two columns of u and v encode user-item biases.
            The second-last column of v is set to all 1's and the last column of u is set to all 1's.
        """
        u = u_init.copy() if u_init is not None else np.random.normal(0, 1, (m, rank + 2))
        v = v_init.copy() if v_init is not None else np.random.normal(0, 1, (n, rank + 2))

        indices = [(i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1]) if mat_ind[i][j] == 1]

        
        mu_global = 0.0 # the global mean of the ratings matrix.
        for i, j in indices:
            mu_global += mat[i][j]
        mu_global = mu_global / len(indices)


        mat = mat.copy() - mu_global # don't modify the passed in copy of the ratings matrix
        
        iters = 0
        stop_condition_met = False
        err_prev = MatrixFactorization.loss_function(u, v, mat, mat_ind, l2_constant, rating_global_mean=0.0)
        vel_u = np.zeros(u.shape)
        vel_v = np.zeros(v.shape)
        while not stop_condition_met and iters < iterations_max:
            iters += 1
            if print_iters:
                print(iters, err_prev, alpha)
            
            random.shuffle(indices)
            for i, j in indices:
                err_ij = mat[i][j] - u[i] @ v[j].transpose()

                # Gradient vectors
                u_grad = l2_constant * u[i,:] - err_ij * v[j]
                v_grad = l2_constant * v[j,:] - err_ij * u[i]

                # Update velocities for Nesterov optimizer
                vel_u[i] = momentum * vel_u[i] - alpha * u_grad
                vel_v[j] = momentum * vel_v[j] - alpha * v_grad
                
                u[i] = u[i] + momentum * vel_u[i] - alpha * u_grad
                v[j] = v[j] + momentum * vel_v[j] - alpha * v_grad
                #u[i] = u[i] - alpha * u_grad
                #v[j] = v[j] - alpha * v_grad
                
                u[i][-1] = 1.0 # last column of u is fixed for the user biases.
                v[i][-2] = 1.0 

            err = MatrixFactorization.loss_function(u, v, mat, mat_ind, l2_constant, rating_global_mean=0.0)
            #if err_prev - err < 0.00001:
            #    stop_condition_met = True
            error_percentage_change = err / err_prev - 1

            # Bold driver algorithm for adjusting the step size.
            if bold_driver and error_percentage_change < 0:
                alpha = 1.05 * alpha
            elif bold_driver and error_percentage_change > 1e-10:
                alpha = alpha / 2
            err_prev = err
            
        print(f"{iters} iterations to converge")
        return (u, v, mu_global) # R = u v + mu_global
    
    @staticmethod
    def factorize(rank, mat, mat_ind, alpha=0.1, alpha_stop=0.0001, iterations_max=1000):
        """ Factorise PartialMatrix mat as U x V^T, where mat is m x n, U is m x k, V is n x k. 
            Uses gradient descent with line search.
        """
        m, n = mat.shape

        u = np.random.normal(0, 1, (m, rank))
        v = np.random.normal(0, 1, (n, rank))

        ind_by_rows = [[j for j in range(mat.shape[1]) if mat_ind[i][j] == 1] for i in range(mat.shape[0])]
        ind_by_cols = [[i for i in range(mat.shape[0]) if mat_ind[i][j] == 1] for j in range(mat.shape[1])]
        
        iters = 0
        stop_condition_met = False
        while not stop_condition_met and iters < iterations_max:
            iters += 1
            print(iters)
            
            u_grad, v_grad = MatrixFactorization.gradient_sparse(rank, u, v, mat, mat_ind, ind_by_rows, ind_by_cols)
            #print(u_grad)
            alpha_c = alpha
            err = MatrixFactorization.mse(u, v, mat, mat_ind)
            while True:
                u_new = u - alpha_c*u_grad
                v_new = v - alpha_c*v_grad
                err_new = MatrixFactorization.mse(u_new, v_new, mat, mat_ind)
                if err_new >= err:
                    if alpha_c < alpha_stop:
                        stop_condition_met = True
                        break
                    alpha_c = alpha_c/2
                else:
                    break
            u = u - alpha_c*u_grad
            v = v - alpha_c*v_grad

        print(f"{iters} iterations to converge")
        return (u, v)

    @staticmethod
    def gradient_sparse(rank, u, v, mat, mat_ind, ind_by_rows, ind_by_cols, very_sparse=False):
        # returns two matrices with the same sizes as u and v (i.e. m x k for u and n x k for v).
                    
        if very_sparse:
            u_grad = np.zeros(u.shape)
            v_grad = np.zeros(v.shape)            
            for a in range(u_grad.shape[0]):
                for j in ind_by_rows[a]:
                    u_grad[a] += err_mat[a][j] * (-1*v[j])
                
            for a in range(v_grad.shape[0]):
                for i in ind_by_cols[a]:
                    v_grad[a] += err_mat[i][a] * (-1*u[i])
        else:
            err_mat = mat - u @ v.transpose() # note this is incorrect on undefined entries

            for i in range(err_mat.shape[0]):
                for j in range(err_mat.shape[1]):
                    if mat_ind[i][j] == 0:
                        err_mat[i][j] = 0
            
            u_grad = err_mat @ (-1*v)
            #v_grad = err_mat.transpose() @ (-1*u)
            v_grad = (-1*u).transpose() @ err_mat
            v_grad = v_grad.transpose()
        
        return (u_grad, v_grad)
    
    @staticmethod
    def gradient(rank, u, v, mat, mat_ind):
        # returns two matrices with the same sizes as u and v (i.e. m x k for u and n x k for v).
        u_grad = np.zeros(u.shape)
        v_grad = np.zeros(v.shape)

        for a in range(u_grad.shape[0]):
            for b in range(u_grad.shape[1]):
                for j in range(mat_ind.shape[1]):
                    if mat_ind[a][j] != 0:
                        e = mat[a][j]
                        for p in range(u_grad.shape[1]):
                            e -= u[a][p] * v[j][p]
                        u_grad[a][b] += e * (-1*v[j][b])

        for a in range(v_grad.shape[0]):
            for b in range(v_grad.shape[1]):
                for i in range(mat_ind.shape[0]):
                    if mat_ind[i][a] != 0:
                        e = mat[i][a]
                        for p in range(u_grad.shape[1]):
                            e -= u[i][p] * v[a][p]
                        v_grad[a][b] += e * (-1*u[i][b])

        return (u_grad, v_grad)


def test_example():
    mat = np.array([[5, 5, 0, 1],
                    [5, 4, 1, 2],
                    [0, 0, 5, 3],
                    [0, 0, 4, 5]])
    """
    mat = np.array([[5, 0, 0, 0],
                [0, 5, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
    """
    mat_ind = np.array([[1, 1, 0, 1],
                        [1, 1, 1, 1],
                        [1, 0, 1, 0],
                        [1, 1, 1, 1]])
    mf = MatrixFactorization
    u, v, mu = mf.sgd_factorize(rank=1, mat=mat, mat_ind=mat_ind)
    print(u)
    print(v)
    print('Global mean:', mu)
    print(u @ v.transpose() + mu)
    print(mf.mse(u, v, mat, mat_ind, mu))
