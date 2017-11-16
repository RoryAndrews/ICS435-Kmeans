#!/usr/bin/env python
"""For testing KMeans"""

import argparse
from itertools import permutations
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def main(args):
    """Main function that runs tests."""

    if args.get('test') == 'converging':
        converging_test(**args)

    # The following will either be removed or cleaned up.
    elif args.get('test') == 'basictest':
        clusterparams = {'n_samples': args.get('samples', 100),
                         'centers': generate3_equidistance_points(distance=10),
                         'cluster_std': args.get('std', 1.0)}
        samples, labels = make_blobs(**clusterparams)
        estimator = KMeans(n_clusters=3, init='k-means++')
        print(samples)
        estimator.fit(samples)
        print(estimator.labels_)
        print(labels)

    elif args.get('test') == 'equiclustergen':
        clusterparams = {'n_samples': 100,
                         'centers': generate3_equidistance_points(),
                         'cluster_std': 1.0}

        for _x in range(10, 6, -1):
            clusterparams['centers'] = generate3_equidistance_points(_x)
            show_cluster_example(clusterparams, colors.ListedColormap(['r', '#00FF00', 'b']))

# runs=100, steps=11, maxdistance=10.0, mindistance=0.0, samples=100, std=1.0
def converging_test(**kwargs):
    """Tests the accuracy of KMeans as clusters approach each other."""
    runs = kwargs.get('runs', 100)
    steps = kwargs.get('steps', 11)
    maxdistance = kwargs.get('maxdistance', 10.0)
    mindistance = kwargs.get('mindistance', 0.0)
    samples = kwargs.get('samples', 100)
    std = kwargs.get('std', 1.0)
    assert steps > 0
    assert maxdistance > mindistance
    assert mindistance >= 0

    clusterparams = {'n_samples': samples,
                     'centers': None,
                     'cluster_std': std}

    estimator = KMeans(n_clusters=3, init='k-means++')
    for _x in np.linspace(maxdistance, mindistance, steps):
        clusterparams['centers'] = generate3_equidistance_points(distance=_x)
        mean, standard_deviation = test_error_rate(estimator, clusterparams, runs=runs)
        print("distance=%f: error-rate=%f +/-%f" % (_x, mean, standard_deviation))


def test_error_rate(estimator, clusterparams, runs=100):
    """This function takes clusterparams and uses it to generate sample data
    and test KMeans over the amount of runs given then returning the average
    error rate."""
    error_rate_list = list()

    # Do multiple runs
    for _x in range(0, runs):
        samples, labels = make_blobs(**clusterparams)
        estimator.fit(samples)

        # Actual and estimated labels won't match so we must map them.
        # Each possibile mapping is tested and the lowest error rate is used.
        # e.g. for mapping, if there are three labels 0,1,2
        # -> mapping = (0,1,2), (0,2,1), (2,1,0), etc.
        # So if estimator label = 1 then we pretend it is the number at index 1 of mapping
        error_rate = np.inf
        for mapping in permutations(range(len(clusterparams['centers']))):
            error = 0
            for label, result in zip(labels, estimator.labels_):
                if label != mapping[result]:
                    error += 1
            result = error/len(samples)
            if result < error_rate:
                error_rate = result
        error_rate_list.append(error_rate)
    # show_result(estimator, samples)
    return np.mean(error_rate_list), np.std(error_rate_list) # return statistics


def show_result(estimator, samples,
                color_map=colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b'])):
    """Displays the resulting labels from KMeans estimator."""
    x_s, y_s = samples.T
    plt.scatter(x=x_s, y=y_s, c=estimator.labels_, cmap=color_map)
    plt.show()


def show_cluster_example(clusterparams,
                         color_map=colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b'])):
    """This generates and displays the clusters using matplotlib."""
    samples, labels = make_blobs(**clusterparams)
    x_s, y_s = samples.T
    plt.scatter(x=x_s, y=y_s, c=labels, cmap=color_map)
    plt.show()


def generate3_equidistance_points(distance=5.0):
    """Returns list of 3 tuples representing 3 equidistant points."""
    return [(0, 0),
            (np.cos(np.pi/3) * distance, np.sin(np.pi/3) * distance),
            (np.cos(np.pi/3) * -distance, np.sin(np.pi/3) * distance)]


def argumentparse():
    """Returns args for main."""
    parser = argparse.ArgumentParser(description='KMeans testing.')
    parser.add_argument('-t', '--test', type=str, metavar='testname',
                        help='The name of the test to be run.')
    parser.add_argument('-r', '--runs', type=int,
                        help='How many runs to go through to get the mean and standard deviation.')
    parser.add_argument('-d', '--std', type=float,
                        help='The standard deviation of the gaussian blobs.')
    parser.add_argument('-s', '--samples', type=int,
                        help='How many samples to a guassian blob.')
    parser.add_argument('--steps', type=int, default=None,
                        help='How many steps in the changing parameters of a test.')
    parser.add_argument('--maxdistance', type=float,
                        help='Max distance for separation of clusters.')
    parser.add_argument('--mindistance', type=float,
                        help='Min distance for separation of clusters.')
    args = parser.parse_args()

    # Returns arguments as dict with empty arguments removed.
    return {k: v for k, v in vars(args).items() if v is not None}

if __name__ == '__main__':
    main(argumentparse())
