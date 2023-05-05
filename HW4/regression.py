import csv
import math
import random

import numpy as np
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        dataset = []
        index = 1
        # Load data
        for row in reader:
            dataset.append(row)
        for row in dataset:
            del row[0]
        del dataset[0]
        for row in dataset:
            i = 0
            for x in row:
                row[i] = float(x)
                i += 1
    return np.asarray(dataset)


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """

    size = len(dataset)
    mean = 0
    standard = 0

    for row in dataset:
        mean += row[col]
    mean = mean / size

    for row in dataset:
        standard += math.pow((row[col] - mean), 2)

    standard = math.sqrt(standard / (size - 1))

    mean = round(mean, 2)
    standard = round(standard, 2)
    print(size)
    print(mean)
    print(standard)
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """

    mse = 0

    for row in range(0, len(dataset)):
        sum = betas[0] - dataset[row][0]
        col = 0
        for i in range(1, len(betas)):
            sum += dataset[row][cols[col]] * betas[i]
            col += 1
        mse += math.pow(sum, 2)

    mse = mse / len(dataset)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []

    for b in range(0, len(betas)):
        sum = 0
        total = 0
        for row in range(0, len(dataset)):
            current = betas[0] - dataset[row][0]
            col = 0
            for i in range(1, len(betas)):
                current += dataset[row][cols[col]] * betas[i]
                col += 1
            if b != 0:
                sum += current * dataset[row][cols[b - 1]]
                sum = round(sum, 8)
            else:
                sum += current
                sum = round(sum, 8)
        grads.append(round(sum * 2 / len(dataset), 8))

    return np.asarray(grads)

    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    # FIX MSE and formatting
    mybetas = betas
    for x in range(1, T + 1):
        out = []
        temp = []
        g = gradient_descent(dataset, cols, mybetas)
        for y in range(0, len(mybetas)):
            mybetas[y] -= g[y] * eta
        out.append(x)
        out.append((round(regression(dataset, cols, mybetas), 2)))
        for i in range(0, len(mybetas)):
            out.append(mybetas[i])
        print(out)

    pass


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    y = []
    x = []
    for row in dataset:
        temp = [1]
        y.append(row[0])
        for col in cols:
            temp.append(row[col])
        x.append(temp)

    array_x = np.array(x)
    array_y = np.array(y)

    transposed = np.transpose(array_x)
    inversed = np.linalg.inv(transposed.dot(array_x))
    betas = inversed.dot(transposed).dot(array_y)

    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    result = betas[1]
    counter = 2
    for feature in features:
        result += betas[counter] * feature
        counter += 1
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linear = []
    for row in X:
        xval = row[0]
        y = betas[0] + (betas[1] * xval) + np.random.normal(loc=0.0, scale=sigma)
        linear.append([y, xval])
    linear_ret = np.asarray(linear)

    quad = []
    for row in X:
        xval = row[0]
        y = alphas[0] + (alphas[1] * math.pow(xval, 2)) + np.random.normal(loc=0.0, scale=sigma)
        quad.append([y, xval])
    quad_ret = np.asarray(quad)

    return linear_ret, quad_ret


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = []
    for y in range(0, 1000):
        X.append([random.randint(-100, 100)])

    betas = []
    alphas = []
    for y in range(0, 2):
        beta = random.random()
        alpha = random.random()
        if beta == 0:
            beta = beta + 1
        if alpha == 0:
            alpha = alpha + 1
        betas.append(beta)
        alphas.append(alpha)

    sigmas = [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000]
    linear_sigs = []
    quad_sigs = []
    for sigma in sigmas:
        set = synthetic_datasets(betas, alphas, X, sigma)
        linear_sigs.append(set[0])
        quad_sigs.append(set[1])

    lin_mse = []
    quad_mse = []
    for linear in linear_sigs:
        lin_mse.append(compute_betas(linear, cols=[1])[0])

    for quad in quad_sigs:
        quad_mse.append(compute_betas(quad, cols=[1])[0])

    plt.plot(sigmas, lin_mse, '-o')
    plt.plot(sigmas, quad_mse, '-o')
    plt.legend(["MSE of Linear Dataset", "MSE of Quadratic Dataset"])

    plt.ylabel("MSE of Trained Model")
    plt.xlabel("Standard Deviation of Error Term")

    plt.yscale("log")
    plt.xscale("log")

    plt.savefig("mse.pdf", format="pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
