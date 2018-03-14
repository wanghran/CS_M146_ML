"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *
import copy

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part c: implement (hint: use np.random.choice)
    initial_points = np.random.choice(points, k)
    return initial_points
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part f: implement
    initial_points = []
    return initial_points
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    k_clusters = ClusterSet()
    if init == 'cheat':
        init_cen = cheat_init(points)
    else:
        init_cen = random_init(points, k)

    for i in range(len(init_cen)):
        cluster = Cluster([init_cen[i]])
        k_clusters.add(cluster)

    for point in points:
        if point in init_cen:
            continue
        dis = []
        for center in init_cen:
            dis.append(np.sqrt((point.attrs[0] - center.attrs[0]) ** 2 +
                               (point.attrs[1] - center.attrs[1]) ** 2))
        k_clusters.members[dis.index(min(dis))].points.append(point)
    prev_clu = ClusterSet()
    for i in range(k):
        prev_clu.add(Cluster([points[0]]))

    while 1:
        print k_clusters.equivalent(prev_clu)

        if k_clusters.equivalent(prev_clu):
            break

        cent = []
        prev_clu = copy.deepcopy(k_clusters)

        for cluster in k_clusters.members:
            cent.append(cluster.centroid())
            cluster.reset()
        for point in points:
            dis = []
            for center in cent:
                dis.append(np.sqrt((point.attrs[0] - center.attrs[0]) ** 2 +
                                   (point.attrs[1] - center.attrs[1])) ** 2)
            k_clusters.members[dis.index(min(dis))].points.append(point)

    if plot:
        plot_clusters(k_clusters, "plot with centroids", k_clusters.centroids())

    return k_clusters
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part e: implement
    k_clusters = ClusterSet()

    return k_clusters
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    
    ### ========== TODO : START ========== ###
    # part d, part e, part f: cluster toy dataset
    np.random.seed(1234)
    points = generate_points_2d(3)
    kMeans(points, 3)
    # a = Point('1', '1', [1.0, 1.0])
    # b = Point('2', '2', [2.0, 2.0])
    # c = Point('3', '3', [3.0, 3.0])
    #
    # clu1 = ClusterSet()
    # clu2 = ClusterSet()
    # clu1.add(Cluster([a, b, c]))
    # clu2.add(Cluster([a, b, c]))
    # print clu2
    # print clu1.equivalent(clu2)
    ### ========== TODO : END ========== ###
    
    


if __name__ == "__main__" :
    main()
