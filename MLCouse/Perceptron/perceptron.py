import pandas as pd
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
plt.style.use("ggplot")

IMG_SHAPE = [28, 28]   


def get_data():
    val = pd.read_csv("validation.csv")
    test = pd.read_csv("test.csv")
    return dict(val=val, test=test)


def show_one(row, label):
    img_label = row[label]
    img = row.drop(label)
    pixels = np.reshape(img.values, IMG_SHAPE)
    fig, ax = plt.subplots()
    ax.imshow(pixels, cmap="bone_r")
    ax.set_title("Label: {0}".format(label))
    plt.show()


def linear_kernel(u, v):
    """
    Computes a the linear kernel K(u, v) = u dot v + 1
    """
    # TODO: Implement the linear kernel function.
    return np.dot(u, v) + 1

def make_polynomial_kernel(d):
    """
    Returns a function which computes the degree-d polynomial kernel between a
    pair of vectors (represented as 1-d Numpy arrays) u and v.
    """
    def kernel(u, v):
        # TODO: Implement the polynomial kernel function.
        return (np.dot(u, v) + 1) ** d

    return kernel    


def exponential_kernel(u, v):
    """
    Computes the exponential kernel between vectors u and v, with sigma = 10.
    """
    sigma = 10
    # TODO: Implement the exponential kernel function.
    return np.exp(np.sqrt(np.sum((u - v) ** 2, axis=1)) / (-2 * sigma * sigma))



def compute_y_hat(x_t, y_mistake, X_mistake, kernel):
    """
    x_t is a vector representing the current training instance.
    y_mistake is a vector of the outputs for all points that the algorithm has
        gotten wrong so far.
    X_mistake is a matrix whose ith row is the ith input that the algorithm got
        wrong so far.
    kernel takes two vectors u, vand returns K(u, v).
    """
    def sign(x): return 1 if x >= 0 else -1
    n_mistake = len(y_mistake)
    if not n_mistake:
        return sign(0)
    else:
        # TODO: Compute y hat.
        return sign(np.sum(kernel(X_mistake, x_t) * y_mistake))


def compute_loss(m):
    """
    Given a boolean mistake vector, compute the losses for T = 100, 200, ...,
    1000.
    """
    loss = np.cumsum(m) / np.arange(1, len(m) + 1)
    return pd.Series(loss[99::100], np.arange(100, 1100, 100))


def fit_perceptron(df, kernel, label):

    y = df[label].values
    X = df.drop(label, axis=1).values.astype(np.float)
    N, D = X.shape
    m = np.repeat(False, N)    
    for t in range(N):
        x_t = X[t, :]              
        y_t = y[t]
        y_mistake = y[m]
        X_mistake = X[m]
        y_hat = compute_y_hat(x_t, y_mistake, X_mistake, kernel)
        if y_hat != y_t:
            m[t] = True        
    loss = compute_loss(m)
    return loss


def run_linear_kernel():
  
    data = get_data()
    val = data["val"]
    loss = fit_perceptron(val, linear_kernel, "valy")
    fig, ax = plt.subplots()
    loss.plot(ax=ax, marker=".")
    ax.set_title("Perceptron with linear kernel\nEvaluated on validation set")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction mistakes")
    fig.savefig("linear-kernel")


def run_polynomial_kernel():
    data = get_data()
    val = data["val"]
    ds = [1, 3, 5, 7, 10, 15, 20]
    losses = [fit_perceptron(val, make_polynomial_kernel(d),"valy") for d in ds]
    losses = pd.concat(losses, axis=1)
    losses.columns = ds
    final_losses = losses.iloc[-1]
    fig, ax = plt.subplots()
    final_losses.plot(ax=ax, marker=".", legend=False)
    ax.set_xlim(0, 21)
    ax.set_title("Perceptron with polynomial kernel\n" +
                 "Evaluated on validation set")
    ax.set_xticks(ds)
    ax.set_xlabel("Degree of kernel")
    ax.set_ylabel("Final fraction mistakes")
    fig.savefig("polynomial-kernel.png")


def run_poly_expon():
    best_degree = 3         
    data = get_data()
    test = data["test"]
    kernel_poly = make_polynomial_kernel(best_degree)
    loss_poly = fit_perceptron(test, kernel_poly,"testy")
    loss_expon = fit_perceptron(test, exponential_kernel,"testy")
    losses = pd.DataFrame(dict(polynomial=loss_poly,
                               expon=loss_expon))
    losses = losses[["polynomial", "expon"]]
    fig, ax = plt.subplots()
    losses.plot(ax=ax, marker=".")
    ax.set_title("Perceptron with polynomial and exponential kernels\n" +
                 "Evaluated on test set")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction mistakes")
    fig.savefig("poly-expon-kernel.png")


def main():
    "Based on command line flag, run one of 3 question subparts"
    flag = argv[1]
    dispatch = dict(linear=run_linear_kernel,
                    poly=run_polynomial_kernel,
                    expon=run_poly_expon)
    if flag not in dispatch:
        print("Not a valid argument to the script.")
    dispatch[flag]()


if __name__ == "__main__":
    main()