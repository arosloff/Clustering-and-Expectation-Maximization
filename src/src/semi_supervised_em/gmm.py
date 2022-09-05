import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group

    np.random.shuffle(x)
    x_split = np.split(x, K)
    mu = np.array(list(map(lambda x: mean(x), x_split)))
    sigma = np.array(list(map(lambda x: covariance(x), x_split)))

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)

    phi = np.ones(K) / K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)

    w = np.ones([x.shape[0], K]) / K

    n = w.shape[0]
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # Just a placeholder for the starter code
        # *** START CODE HERE
        prev_ll = ll

        dim = x.shape[1]

        # (1) E-step: Update your estimates in w
        w = eStep(w, dim, sigma, mu, x, phi)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = phiUpdate(w)

        for j in range(w.shape[1]):
            mu[j] = muJUpdate(w[:, j], x)

        for j in range(w.shape[1]):
            sigma[j] = sigJUpdate(w[:, j], mu[j], x)

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        logLikeSum = 0
        # for i in range (w.shape[0]):
        #     for j in range (w.shape[1]):
        #         logLikeSum += w[i][j] * np.log((multiG(dim, sigma[j], mu[j], x[i], phi[j])) / w[i][j])
        for i in range(w.shape[0]):
            probSum = 0
            for j in range(w.shape[1]):
                probSum += multiG(dim, sigma[j], mu[j], x[i], phi[j])
            logLikeSum += np.log(probSum)
        ll = logLikeSum
        if it > 0:
            if ll < prev_ll:
                print("OH GOD OH NO")
        it += 1
    print(f"unsupervised iterations: {it}")
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        prev_ll = ll
        dim = x.shape[1]
        # (1) E-step: Update your estimates in w

        w = eStep(w, dim, sigma, mu, x, phi)

        # (2) M-step: Update the model parameters phi, mu, and sigma

        phi = phiUpdate(w, True, alpha, z_tilde)
        #print(phi)

        for j in range(w.shape[1]):
            mu[j] = ssMujUpdate(j, w[:, j], x, x_tilde, alpha, z_tilde)

        for j in range(w.shape[1]):
            sigma[j] = ssSigJUpdate(j, w[:, j], mu[j], x, x_tilde, alpha, z_tilde)

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        ulogLikeSum = 0

        for i in range(x.shape[0]):
            probSum = 0
            for j in range(w.shape[1]):
                probSum += multiG(dim, sigma[j], mu[j], x[i], phi[j])
            ulogLikeSum += np.log(probSum)

        slogLikeSum = 0

        for i in range(x_tilde.shape[0]):
            probSum = 0
            for j in range(w.shape[1]):
                if z_tilde[i] == j:
                    probSum += multiG(dim, sigma[j], mu[j], x_tilde[i], phi[j])
            slogLikeSum += np.log(probSum)

        ll = ulogLikeSum + (alpha * slogLikeSum)
        if it > 0:
            if ll < prev_ll:
                print("oh no")

        # for i in range(w.shape[0]):
        #     probSum = 0
        #     for j in range(w.shape[1]):
        #         probSum += multiG(dim, sigma[j], mu[j], x[i], phi[j])
        #     logLikeSum += np.log(probSum)
        # ll = logLikeSum
        it += 1
    print(f"supervised iterations: {it}")
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Define any helper functions
def mean(gaussian):
    return np.mean(gaussian, axis=0)

def covariance(gaussian):
    return np.cov(np.transpose(gaussian))

def multiG(dim, sigmaj, muj, xi, phij):
    sqrtDet = np.linalg.det(sigmaj) ** (0.5)
    piTerm = ((2*np.pi) ** (dim/2)) * sqrtDet
    invPiTerm = 1 / piTerm

    invSig = np.linalg.inv(sigmaj)
    meanDiffDotSig = np.dot(np.transpose(xi - muj), invSig)
    quadratic = np.dot(meanDiffDotSig, (xi - muj))
    expTerm = np.exp(-0.5 * quadratic)
    #print((invPiTerm * expTerm * phij))
    return (invPiTerm * expTerm * phij)

def phiUpdate(w, ss = False, alpha = 1.0, z = None):
    # This one is going to just do it all at the same time by adding up the columns for non SS
    if not ss:
        return np.sum(w, axis=0) / w.shape[0]
    else:
        phi = np.zeros(w.shape[1])
        for j in range(w.shape[1]):
            wSum = 0
            zSum = 0
            for i in range(w.shape[0]):
                wSum += w[i][j]
            for i in range(z.shape[0]):
                if z[i] == j:
                    zSum += 1
            phi[j] = (wSum + (alpha * zSum)) / (w.shape[0] + (alpha * z.shape[0]))
        #print(phi)
        return phi

def muJUpdate(wj, x):
    # Called iteratively, sadly
    num = 0
    den = 0
    for i in range(wj.shape[0]):
        num += wj[i] * x[i]
        den += wj[i]
    return (num/den)

def ssMujUpdate(j, wj, x, x_tilde, alpha, z):
    wSum = 0
    wxSum = 0
    xTildeSum = 0
    zSum = 0
    for i in range(wj.shape[0]):
        wxSum += wj[i] * x[i]
        wSum += wj[i]
    for i in range(x_tilde.shape[0]):
        if z[i] == j:
            zSum += 1
            xTildeSum += x_tilde[i]

    return ((wxSum + (alpha * xTildeSum)) / (wSum + (alpha * zSum)))


def sigJUpdate(wj, muj, x):
    num = 0
    den = 0
    for i in range(wj.shape[0]):
        diff = x[i] - muj
        num += wj[i] * (np.outer(diff, diff))
        den += wj[i]
    return (num/den)

def ssSigJUpdate(j, wj, muj, x, x_tilde, alpha, z):
    xOutSum = 0
    xTSum = 0
    wSum = 0
    zSum = 0
    for i in range(wj.shape[0]):
        diff = x[i] - muj
        xOutSum += wj[i] * (np.outer(diff, diff))
        wSum += wj[i]
    for i in range (x_tilde.shape[0]):
        if z[i] == j:
            diff = x_tilde[i] - muj
            xTSum += np.outer(diff, diff)
            zSum += 1
    return (xOutSum + (alpha * xTSum)) / (wSum + (alpha * zSum))


def eStep(w, dim, sigma, mu, x, phi):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            num = multiG(dim, sigma[j], mu[j], x[i], phi[j])
            den = 0
            for k in range(w.shape[1]):
                den += multiG(dim, sigma[k], mu[k], x[i], phi[k])
            w[i][j] = (num/den)

    return w

# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        #main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
