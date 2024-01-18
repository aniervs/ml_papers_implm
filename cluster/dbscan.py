# PAPER: https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons 
from sklearn.preprocessing import StandardScaler

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
        self.labels = [None for _ in range(n)] # None means unlabeled, -1 means noise
        self.n_clusters = 0

        for i in range(n):
            if self.labels[i] is None and self._expand_cluster(X, i):
                self.n_clusters += 1
                        
                
    def fit_predict(self, X):
        self.fit(X)
        return self.labels    
        
    def _region_query(self, X, p):
        """
        p: a point
        eps: threshold on the neighborhood distance

        return: list of ids of points at distance < eps from p

        (done with Brute Force now)
        """
        
        n = len(X)
        result_ids = [i for i in range(n) if distance(X[i], p, self.metric) < self.eps]
        return result_ids

    def _expand_cluster(self, X, p_idx):
        """
        p_idx: id of the point candidate to be core of a cluster
        
        return: True/False whether p_idx could expand to a cluster or not
        
        Algorithm:
        This function runs a BFS to find the cluster defined by p_idx
        """
        
        seeds = self._region_query(X, X[p_idx])
        if len(seeds) < self.min_pts:
            self.labels[p_idx] = -1 # Noise
            return False 
        
        for q_idx in seeds:
            self.labels[q_idx] = self.n_clusters
            
        i = 0
        while i < len(seeds):
            p_idx = seeds[i]
            p = X[p_idx]
            neighborhood = self._region_query(X, p)
            if len(neighborhood) >= self.min_pts:    
                for q_idx in neighborhood:
                    if self.labels[q_idx] is None or self.labels[q_idx] == -1:
                        if self.labels[q_idx] == -1:
                            seeds.append(q_idx)
                        self.labels[q_idx] = self.n_clusters
            i += 1
        return True 
                        

def test():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    X = StandardScaler().fit_transform(X)
    
    
    dbscan = DBSCAN(eps=1, min_pts=5)
    labels = dbscan.fit_predict(X)
    
    print()
    print("Number of noise points:", labels.count(-1))
    print("Number of clusters:", dbscan.n_clusters)
    print()
    
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

if __name__ == '__main__':
    test()