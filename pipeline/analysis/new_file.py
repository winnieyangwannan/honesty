
from sklearn.cluster import k_means
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris
from validclust import dunn
data = load_iris()['data']
_, labels, _ = k_means(data, n_clusters=3)
dist = pairwise_distances(data)
dunn(dist, labels)