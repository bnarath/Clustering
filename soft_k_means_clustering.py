import numpy as np
import matplotlib.pyplot as plt

def cost(X,R,M):
    cost=0
    for k in range(len(M)):
        for n in range(len(X)):
           cost+=R[n,k]*d(M[k], X[n])
    return cost



def d(u,v):
    diff = u - v
    return diff.dot(diff)

def plot_k_means(X, K, max_iter=20, beta=1.0):

    N,D = X.shape #N samples, D features
    M = np.zeros((K,D)) #K clusters, M - cluster centre
    R = np.zeros((N, K)) #Responsibility

    for k in range(K):
       M[k] = X[np.random.choice(N)] #Random K values of X are chosen as cluster centroids
       costs = np.zeros(max_iter)
    for i in range(max_iter):
        for k in range(K): #Measure responsibility
            for n in range(N):
                R[n,k] = np.exp(-beta*d(M[k], X[n])) / np.sum([np.exp(-beta*d(M[j], X[n])) for j in range(K)])

        for k in range(K):#Centroid calculation
            M[k] = R[:,k].dot(X)/R[:,k].sum()

        costs[i] = cost(X, R, M) #Cost calculation

        if i > 0:
            if np.abs(costs[i]-costs[i-1]) < 0.1:
                break

    plt.plot(costs)
    plt.title("Costs")
    plt.show()

    random_colors = np.random.random((K,3)) #Similar X's based on their responsibility will get similar color
    colors = R.dot(random_colors)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()

def main():
    D=2
    s=4
    mu1 = np.array([0,0])
    mu2 = np.array([s,s])
    mu3 = np.array([0,s])

    N=900
    X=np.zeros((N,D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600,:] = np.random.randn(300, D) + mu2
    X[600:,:] = np.random.randn(300, D) + mu3

    plt.scatter(X[:,0], X[:,1])
    plt.show()

    K=3
    plot_k_means(X,K)

    K=5
    plot_k_means(X,K, max_iter=30)

    K=5
    plot_k_means(X,K, max_iter=30, beta=0.3)

if __name__ =='__main__':
    main()
