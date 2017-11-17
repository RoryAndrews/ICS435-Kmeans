#!/usr/bin/env python
"""For testing KMeans"""

import argparse
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as anim
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def main(args):
    """Main function that runs tests."""
    print(args)

    if args.get('test') == 'converging':
        converging_test(**args)

    elif args.get('test') == 'skewing':
        skewing_test(**args)

    else:
        print("No test found for \"%s\"" % args.get('test'))



def converging_test(**kwargs):
    """Tests the accuracy of KMeans as clusters approach each other."""
    runs = kwargs.get('runs', 100)
    steps = kwargs.get('steps', 11)
    maxdistance = kwargs.get('maxdistance', 10.0)
    mindistance = kwargs.get('mindistance', 0.0)
    n_samples = kwargs.get('samples', 100)
    std = kwargs.get('std', 1.0)
    assert maxdistance > mindistance
    assert mindistance >= 0

    results = list()

    clusterparams = {'n_samples': n_samples,
                     'centers': None,
                     'cluster_std': std}

    estimator = KMeans(n_clusters=3, init='k-means++')
    for _x in np.linspace(maxdistance, mindistance, steps):
        clusterparams['centers'] = generate3_equidistance_points(distance=_x)
        _r = test_error_rate(estimator, clusterparams, runs=runs)
        _r['var']=_x
        results.append(_r)

        print("distance=%f: error-rate=%f +/-%f" % (_x, _r['mean'], _r['std']))

    # Save results to gif
    if kwargs.get('gif') is not None:
        filename = kwargs.get('gif')
        if filename == 'default':
            filename = 'converging_test.gif'
        fig = plt.figure()
        gifwriter = anim.ImageMagickWriter(fps=1)
        gifwriter.setup(fig, filename, dpi=150)

        for result in results:
            x_s, y_s = result['samples'].T
            plt.scatter(x=x_s, y=y_s, c=result['elabels'],
                        cmap=colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b']))
            x_s, y_s = result['clusters'].T
            plt.scatter(x=x_s, y=y_s, c=range(len(x_s)), marker='x', s=100,
                        cmap=colors.ListedColormap(['k']))
            plt.axis(1.2 * maxdistance * np.array([-1, 1, -0.3, 1]))
            plt.xlabel("distance=%f: error-rate=%f +/-%f" % (result['var'], result['mean'], result['std']))
            plt.title('Convergence Test')
            gifwriter.grab_frame()
            plt.clf()
        gifwriter.finish()
        print("saved %s" % filename)

    if kwargs.get('graph') is not None:
        filename = kwargs.get('graph')
        if filename == 'default':
            filename = 'converging_test.png'
        fig = plt.figure()
        for result in results:
            plt.errorbar(result['var'], result['mean'], yerr=result['std'], fmt='ob', capsize=5)
        plt.xlabel('distance')
        plt.ylabel('error rate')
        plt.title('Convergence Test')
        plt.gca().invert_xaxis()
        plt.savefig(filename, dpi=150)
        print("saved %s" % filename)



def skewing_test(**kwargs):
    """Tests the accuracy of KMeans as gaussian blobs are skewed in the y axis.
    std for y-axis = std + (var_rate * step)"""
    runs = kwargs.get('runs', 100)
    steps = kwargs.get('steps', 20)
    n_samples = kwargs.get('samples', 200)
    variance = kwargs.get('variance', 20.0)
    std = kwargs.get('std', 1.0)
    maxdistance = kwargs.get('maxdistance', 20.0)

    results = list()

    clusterparams = {'maxdistance': maxdistance,
                     'n_samples': n_samples,
                     'std_x': std,
                     'std_y': None}

    estimator = KMeans(n_clusters=3, init='k-means++')
    for _x in np.linspace(1, variance, steps):
        clusterparams['std_y'] = std * _x
        _r = test_error_rate(estimator, clusterparams, gentype='skewed', runs=runs)
        _r['var'] = _x
        results.append(_r)

        print("std=%f: error-rate=%f +/-%f" % (_x, _r['mean'], _r['std']))

    # Save results to gif
    if kwargs.get('gif') is not None:
        filename = kwargs.get('gif')
        if filename == 'default':
            filename = 'skewing_test.gif'
        fig = plt.figure()
        gifwriter = anim.ImageMagickWriter(fps=1)
        gifwriter.setup(fig, str(1) + filename, dpi=150)

        # Estimated results.
        for result in results:
            x_s, y_s = result['samples'].T
            plt.scatter(x=x_s, y=y_s, c=result['elabels'],
                        cmap=colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b']))
            x_s, y_s = result['clusters'].T
            plt.scatter(x=x_s, y=y_s, c=range(len(x_s)), marker='x', s=100,
                        cmap=colors.ListedColormap(['k']))
            plt.xlabel("y standard dev.=%f: error-rate=%f +/-%f" % (result['var'], result['mean'], result['std']))
            plt.title('Skew Test')
            plt.axis('equal')
            gifwriter.grab_frame()
            plt.clf()
        gifwriter.finish()
        print("saved %s" % str(1) + filename)

        fig = plt.figure()
        gifwriter = anim.ImageMagickWriter(fps=1)
        gifwriter.setup(fig, str(2) + filename, dpi=150)
        # Actual labels.
        for result in results:
            x_s, y_s = result['samples'].T
            plt.scatter(x=x_s, y=y_s, c=result['labels'],
                        cmap=colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b']))
            plt.xlabel("y standard dev.=%f: error-rate=%f +/-%f" % (result['var'], result['mean'], result['std']))
            plt.title('Skew Test')
            plt.axis('equal')
            gifwriter.grab_frame()
            plt.clf()
        gifwriter.finish()
        print("saved %s" % str(2) + filename)

    if kwargs.get('graph') is not None:
        filename = kwargs.get('graph')
        if filename == 'default':
            filename = 'skewing_test.png'
        fig = plt.figure()
        for result in results:
            plt.errorbar(result['var'], result['mean'], yerr=result['std'], fmt='ob', capsize=5)
        plt.xlabel('y standard dev.')
        plt.ylabel('error rate')
        plt.title('Sigma Skew Test')
        plt.savefig(filename, dpi=150)
        print("saved %s" % filename)


def test_error_rate(estimator, clusterparams, gentype='default', runs=100):
    """This function takes clusterparams and uses it to generate sample data
    and test KMeans over the amount of runs given then returning the average
    error rate."""
    error_rate_list = list()

    # Do multiple runs
    for _x in range(runs):
        if gentype == 'default':
            samples, labels = make_blobs(**clusterparams)
        elif gentype == 'skewed':
            samples, labels = make_skewed_blobs(**clusterparams)
        estimator.fit(samples)

        # Actual and estimated labels won't match so we must map them.
        # Each possibile mapping is tested and the lowest error rate is used.
        # e.g. for mapping, if there are three labels 0,1,2
        # -> mapping = (0,1,2), (0,2,1), (2,1,0), etc.
        # So if estimator label = 1 then we pretend it is the number at index 1 of mapping
        error_rate = np.inf
        for mapping in permutations(range(len(estimator.cluster_centers_))):
            error = 0
            for label, result in zip(labels, estimator.labels_):
                if label != mapping[result]:
                    error += 1
            result = error/len(samples)
            if result < error_rate:
                error_rate = result
        error_rate_list.append(error_rate)

    mean = np.mean(error_rate_list)
    standard_deviation = np.std(error_rate_list)
    return {'mean': mean,
            'std': standard_deviation,
            'elabels': estimator.labels_,
            'labels': labels,
            'samples': samples,
            'clusters': estimator.cluster_centers_}


def show_result(estimator, samples,
                color_map=colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b'])):
    """Displays the resulting labels from KMeans estimator."""
    x_s, y_s = samples.T
    plt.scatter(x=x_s, y=y_s, c=estimator.labels_, cmap=color_map)
    plt.show()


def show_cluster_example(samples, labels,
                         color_map=colors.ListedColormap(['r', 'y', 'g', '#00FF00', 'b'])):
    """This generates and displays the clusters using matplotlib."""
    x_s, y_s = samples.T
    plt.scatter(x=x_s, y=y_s, c=labels, cmap=color_map)
    plt.axis('equal')
    plt.show()


def make_skewed_blobs(maxdistance, std_x, std_y, n_samples):
    """Makes skewed blobs."""
    centers = (maxdistance * 2 * np.random.random((3, 2))) - maxdistance
    samples_x, labels = make_blobs(n_samples=n_samples,
                                   cluster_std=std_x, centers=centers[:, [0]])
    samples_y = list()
    for label in labels:
        samples_y.append(centers[label][1] + (std_y * np.random.randn()))
    return np.concatenate((samples_x, np.array([samples_y]).T), axis=1), labels


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
    parser.add_argument('--variance', type=float, metavar='var',
                        help='Variance used by some tests.')
    parser.add_argument('--gif', type=str, metavar='filename', default='default',
                        help='Save gif of test with this name.')
    parser.add_argument('--fps', type=int,
                        help='Save gif of test with this fps (if --save is used).')
    parser.add_argument('--graph', type=str, metavar='filename', default='default',
                        help='Save graph of test results.')
    args = parser.parse_args()

    # Returns arguments as dict with empty arguments removed.
    return {k: v for k, v in vars(args).items() if v is not None}

if __name__ == '__main__':
    main(argumentparse())
