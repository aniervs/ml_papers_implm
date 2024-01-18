cluster_id = [None] # tells for each point, its cluster_id
points = [None] # array of points
eps = 1e-3 # change this
min_pts = 1 # change this

def distance(a, b):
    # euclidean
    result = 0
    k = len(a)
    for i in range(k):
        result += (a[i] - b[i])**2
    return result 

def region_query(p):
    """
    p: a point
    eps: threshold on the neighborhood distance
    
    return: list of ids of points at distance < eps from p
    
    (done with Brute Force now)
    # TODO: change later to R*-trees
    """
    result_ids = []
    n = len(points)
    
    for i in range(n):
        if distance(points[i], p) < eps:
            result_ids.append(i)
    return result_ids  

def expand_cluster(p_idx, n_clusters):
    """
    p_idx: id of the point candidate to be core of a cluster
    n_clusters: number of clusters so far. It's also the id of the current cluster
    eps: threshold on the neighborhood distance
    min_pts: threshold on the number of neighbors to consider in order to be "core"
    
    return: True/False whether p_idx could expand to a cluster
    
    Algorithm:
    This function runs a BFS to find the cluster defined by p_idx
    
    """
    
    seeds = region_query(points[p_idx], eps)
    if len(seeds) < min_pts:
        cluster_id[p_idx] = -1
        return False 
    for q_idx in seeds:
        cluster_id[q_idx] = n_clusters
        
    i = 0
    while i < len(seeds):
        p_idx = seeds[i]
        p = points[p_idx]
        neighborhood = region_query(points, p, eps)
        if len(neighborhood) < min_pts:
            continue 
        
        for q_idx in neighborhood:
            if cluster_id[q_idx] is None or cluster_id[q_idx] == -1:
                if cluster_id[q_idx] == -1:
                    seeds.append(q_idx)
                cluster_id[q_idx] = n_clusters
    return True 
                    

def dbscan():
    global cluster_id
    n_clusters = 0
    n = len(points)
    
    cluster_id = [None for _ in range(n)] # None: unclassified, -1: Noise
    
    for i in range(n):
        if cluster_id[i] is None and expand_cluster(points, i, n_clusters, eps, min_pts):
            n_clusters += 1
            
        