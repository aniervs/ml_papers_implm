# PAPER: https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons 
from sklearn.preprocessing import StandardScaler
from collections import deque
    

def distance(p1, p2, metric='euclidean'):
    """
    p1, p2: points
    metric: distance metric
    
    return: distance between p1 and p2
    """
    if metric == 'euclidean':
        return np.sqrt(np.sum((p1 - p2)**2))
    elif metric == 'manhattan':
        return np.sum(np.abs(p1 - p2))
    elif metric == 'cosine':
        return 1 - np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    else:
        raise ValueError('Unknown metric')
    
def region_query(X, p, eps, metric='euclidean'):
    """
    X: dataset (n x d)
    p: point (d)
    eps: epsilon (float)
    metric: distance metric (str)
    
    return: all points in X within eps of p
    (done with Brute Force now)
    """
    neighbors = [i for i in range(X.shape[0]) if distance(X[i], p, metric) < eps]
    return neighbors

class DBSCAN:
    def __init__(self, eps, min_pts, metric='euclidean') -> None:
        self.eps = eps 
        self.min_pts = min_pts
        self.metric = metric
        
    def fit(self, X):
        """
        X: array of points
        """
        n = len(X)
        
        self.neighborhoods = [region_query(X, X[i], self.eps, self.metric) for i in range(n)]
        self.labels = np.array([-1 for _ in range(n)]) # by default everything is Noise
        self.is_core = np.array([len(self.neighborhoods[i]) >= self.min_pts for i in range(n)])
        
        self.n_clusters = 0
        
        for i in range(n):
            if self.is_core[i] and self.labels[i] == -1:
                self._bfs(i)
                self.n_clusters += 1
                
        return self
                        
                
    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def _bfs(self, p_id):
        """
        p_id: id of the core point taken to expand on the current cluster
        
        Algorithm:
        This function runs a BFS to find the cluster defined by p_id
        """
        
        q = deque()
        q.append(p_id)
        self.labels[p_id] = self.n_clusters

        while len(q) > 0:
            p_id = q.popleft() 
            neighbors = self.neighborhoods[p_id]
            for q_id in neighbors:
                if self.labels[q_id] == -1:
                    self.labels[q_id] = self.n_clusters
                    if self.is_core[q_id]:
                        q.append(q_id)
                        
def run_blobs(ax):
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    X = StandardScaler().fit_transform(X)
    
    dbscan = DBSCAN(eps=0.3, min_pts=10)
    labels = dbscan.fit_predict(X)
    
    print()
    print("BLOBS Number of noise points:", np.sum(labels == -1))
    print("BLOBS Number of clusters:", dbscan.n_clusters)
    print()
    
    ax.set_title('Blobs')
    
    for c_id in range(dbscan.n_clusters):
        ax.scatter(X[labels == c_id, 0], X[labels == c_id, 1], label=f'Cluster {c_id}', c=f'C{c_id}')    
    ax.scatter(X[labels == -1, 0], X[labels == -1, 1], label='Noise', c='black')
    ax.legend()
    

def run_moons(ax):
    X, _ = make_moons(n_samples=750, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X)
    
    dbscan = DBSCAN(eps=0.3, min_pts=10)
    labels = dbscan.fit_predict(X)
    
    print()
    print("MOONS Number of noise points:", np.sum(labels == -1))
    print("MOONS Number of clusters:", dbscan.n_clusters)
    print()
    
    ax.set_title('Moons')
    
    for c_id in range(dbscan.n_clusters):
        ax.scatter(X[labels == c_id, 0], X[labels == c_id, 1], label=f'Cluster {c_id}', c=f'C{c_id}')    
    ax.scatter(X[labels == -1, 0], X[labels == -1, 1], label='Noise', c='black')
    ax.legend()

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    run_blobs(ax1)
    run_moons(ax2)
    
    plt.show()

if __name__ == '__main__':
    main()
