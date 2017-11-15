import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def main():
    """Main function that runs tests."""

    # Generating data:
    clusterparams_random = {'n_samples': 100,
                            'centers': 4,
                            'cluster_std': 1.0,
                            'center_box': (-10.0, 10.0)}
    clusterparams = {'n_samples': 100,
                     'centers': [(0, 0), (1, 1), (-1, 1)],
                     'cluster_std': 1.0}


    show_cluster_example(clusterparams_random)
    show_cluster_example(clusterparams)


def test_kmeans(clusterparams):
    """This function takes clusterparams and uses it to generate sample data and test KMeans."""

    samples, labels = make_blobs(**clusterparams)

def show_cluster_example(clusterparams):
    """This generates and displays the clusters using matplotlib."""
    color_map = colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b'])
    samples, labels = make_blobs(**clusterparams)
    x_s, y_s = samples.T
    plt.scatter(x=x_s, y=y_s, c=labels, cmap=color_map)
    plt.show()

if __name__ == '__main__':
    main()
