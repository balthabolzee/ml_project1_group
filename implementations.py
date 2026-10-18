import numpy as np
from helpers import batch_iter




def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with gradient descent. Returns (w, loss)."""

    w = initial_w

    for n_iter in range(max_iters):

        e = y - tx @ w
        mse = 1 / (2*len(e)) * e.T @ e
        grad_mse = - 1 / len(e) * tx.T @ e
        
        w = w - gamma * grad_mse

    loss = mse

    return w, loss






def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with SGD (batch size 1). Returns (w, loss)."""

    w = initial_w

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):

            e = y - tx @ w
            mse = 1 / (2*len(e)) * e.T @ e

            stochastic_e = y_batch - tx_batch @ w
            grad_mse = - 1 / len(stochastic_e) * tx.T @ stochastic_e
            
            w = w - gamma * grad_mse

    loss = mse

    return w, loss









def least_squares(y, tx):
    """Least squares via normal equations (no np.linalg.lstsq)."""
    raise NotImplementedError

def ridge_regression(y, tx, lambda_):
    """Ridge regression via normal equations. Loss without penalty."""
    raise NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with GD, y in {0,1}. Returns (w, loss)."""
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression. Loss without penalty."""
    raise NotImplementedError