import random
import math

# TODO: Select the first k points from your data set as starting centers.
def init_centers_first_k(data_set, k):
    """
    Initialize centers as first k data points in the data_set.

    Args:
        data_set: a list of data points, each data point is a dictionary:
            {'country': 'country_name',
             'vals': [x_1, ..., x_n]}
        k: the number of mean/clusters.

    Returns:
        centers: a list of k elements: centers initialized using first k data points in your data_set.
                 Each center is a list of numerical values. i.e., 'vals' of a data point.
    """
    centers = []
    for i in range(k):
    	centers.append(data_set[i]['vals'])
    return centers


# TODO: Select k points randomly from your data set as starting centers.
def init_centers_random(data_set, k):
    """
    Initialize centers by selecting k random data points in the data_set.

    Args:
        data_set: a list of data points, each data point is a dictionary:
            {'country': 'country_name',
             'vals': [x_1, ..., x_n]}
        k: the number of mean/clusters.

    Returns:
        centers: a list of k elements: centers initialized using random k data points in your data_set.
                 Each center is a list of numerical values. i.e., 'vals' of a data point.
    """
    centers = []
    countries = []
    while len(centers) < k:
    	ran = random.choice(data_set)
    	if ran['country'] not in countries:
    		centers.append(ran['vals'])
    		countries.append(ran['country'])
    return centers


# TODO: compute the euclidean distance from a data point to the center of a cluster
def dist(vals, center):
    """
    Helper function: compute the euclidean distance from a data point to the center of a cluster

    Args:
        vals: a list of numbers (i.e. 'vals' of a data_point)
        center: a list of numbers, the center of a cluster.

    Returns:
         d: the euclidean distance from a data point to the center of a cluster
    """
    d = 0
    for i in range(len(center)):
    	d = d + (vals[i] - center[i])**2
    return math.sqrt(d)


# TODO: return the index of the nearest cluster
def get_nearest_center(vals, centers):
    """
    Assign a data point to the cluster associated with the nearest of the k center points.
    Return the index of the assigned cluster.

    Args:
        vals: a list of numbers (i.e. 'vals' of a data point)
        centers: a list of center points.

    Returns:
        c_idx: a number, the index of the center of the nearest cluster, to which the given data point is assigned to.
    """
    c_idx = 0
    min_dist  = dist(vals['vals'], centers[0])
    for i in range(1,len(centers)):
    	if dist(vals['vals'], centers[i]) < min_dist:
    		c_idx = i
    		min_dist = dist(vals['vals'], centers[i])
    return c_idx


# TODO: compute element-wise addition of two vectors.
def vect_add(x, y):
    """
    Helper function for recalculate_centers: compute the element-wise addition of two lists.
    Args:
        x: a list of numerical values
        y: a list of numerical values

    Returns:
        s: a list: result of element-wise addition of x and y.
    """
    s = [0 for i in range(len(x))]
    for i in range(len(y)):
    	s[i] = x[i] + y[i]
    return s


# TODO: averaging n vectors.
def vect_avg(s, n):
    """
    Helper function for recalculate_centers: Averaging n lists.
    Args:
        s: a list of numerical values: the element-wise addition over n lists.
        n: a number, number of lists

    Returns:
        s: a list of numerical values: the averaging result of n lists.
    """
    
    total = s[0]['vals']
    avg = [0 for i in range(len(total))]
    for i in range(1, len(s)):
    	total =vect_add(total, s[i]['vals'])
    for i in range(len(total)):
    	avg[i] = total[i] / n
    return avg


# TODO: return the updated centers.
def recalculate_centers(clusters):
    """
    Re-calculate the centers as the mean vector of each cluster.
    Args:
         clusters: a list of clusters. Each cluster is a list of data_points assigned to that cluster.

    Returns:
        centers: a list of new centers as the mean vector of each cluster.
    """
    centers = []
    for i in range(len(clusters)):
    	n = len(clusters[i])
    	center = vect_avg(clusters[i], n)
    	centers.append(center)
    return centers


# TODO: run kmean algorithm on data set until convergence or iteration limit.
def train_kmean(data_set, centers, iter_limit):
    """
    Args:
        data_set: a list of data points, each data point is a dictionary of
            {'country': 'country_name',
             'vals': [x_1, ..., x_n]}
        centers: a list of initial centers.
        iter_limit: a number, iteration limit

    Returns:
        centers: a list of updates centers/mean vectors.
        clusters: a list of clusters. Each cluster is a list of data points.
        num_iterations: a number, num of iteration when converged.
    """
    clusters = [[] for x in range(len(centers))]
    num_iterations = 0
    for i in range(iter_limit):
    	num_iterations += 1
    	clusters = [[] for x in range(len(centers))]
    	for j in range(len(data_set)):
    		c_idx = get_nearest_center(data_set[j], centers)
    		clusters[c_idx].append(data_set[j])
    	old_centers = centers
    	centers = recalculate_centers(clusters)
    	if centers == old_centers:
    		break

    return centers, clusters, num_iterations


# TODO: helper function: compute within group sum of squares
def within_group_ss(cluster, center):
    """
    For each cluter, compute the sum of squares of euclidean distance
    from each data point in the cluster to the empirical mean of this cluster.
    Please note that the euclidean distance is squared in this function.

    Args:
        cluster: a list of data points.
        center: the center for the given cluster.

    Returns:
        ss: a number, the within cluster sum of squares.
    """
    ss = 0.0
    for i in range(len(cluster)):
    	ss = ss + dist(cluster[i]['vals'], center)**2
    return ss


# TODO: compute sum of within group sum of squares
def sum_of_within_group_ss(clusters, centers):
    """
    For total of k clusters, compute the sum of all k within_group_ss(cluster).

    Args:
        clusters: a list of clusters.
        centers: a list of centers of the given clusters.

    Returns:
        sss: a number, the sum of within cluster sum of squares for all clusters.
    """
    sss = 0.0
    for i in range(len(clusters)):
    	sss = sss + within_group_ss(clusters[i], centers[i])
    return sss
