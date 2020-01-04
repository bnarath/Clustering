import numpy as np
from soft_k_means_clustering import plot_k_means


def donut():  # Donut data creation
    N = 1000
    D = 2
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(N // 2) + R_inner  # Randn is std normal
    theta = 2 * np.pi * np.random.random(N // 2)  # Random is uniform in [0.0, 1.0)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer  # Randn is std normal
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    return X


def main():
    X = donut()
    plot_k_means(X, 2)
    X = np.zeros((1000, 2))
    X[:500, :] = np.random.multivariate_normal([0, 0], [[1, 0], [0, 20]], 500)
    X[500:, :] = np.random.multivariate_normal([5, 0], [[1, 0], [0, 20]], 500)
    plot_k_means(X, 2)
    X = np.zeros((1000, 2))
    X[:950, :] = np.array([0, 0]) + np.random.randn(950, 2)
    X[950:, :] = np.array([3, 0]) + np.random.randn(50, 2)
    plot_k_means(X, 2)


if __name__ == '__main__':
    main()

#Disadvantages of K-means
#1. Can tackle only spherical form of data not elliptical (donut)
#2. Cannot deal with density
#3. Need to choose K
#4. Sensitive to initialization