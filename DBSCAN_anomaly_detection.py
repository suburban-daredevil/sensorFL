import numpy as np
import itertools
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as shs

'''
Anomaly Detection using DBSCAN Algorithm
Rule of thumb for min_samples = dimensionality + 1
Range of epsilon is from 0.01 to 1 spanning for 20 values
Range of min_samples is from 2 to 25 spanning for 23 values
Total number of combinations = 23 * 20 - 460
'''

'''
https://medium.com/@revag2014/dbscan-an-easy-clustering-algorithm-and-also-how-to-optimize-it-using-grid-search-69a382b63e85
'''

def GridSearchHelper(df):
    eps = np.linspace(0.01, 1, num = 20)  # total of 20 values of eps
    print('Values of epsilon:', eps)

    min_samples = np.arange(2, 25, step = 1)  # max of min_samples = 2 * dimension of df - total of 23 values for min_samples
    print('Min samples values:', min_samples)

    combinations = list(itertools.product(eps, min_samples))
    N = len(combinations)
    print(combinations)
    print('Total Number of combinations:', N)

    scores = []
    all_labels = []

    for i, (eps, min_samples) in enumerate(combinations):
        print('Epsilon:', eps, 'Min samples:', min_samples)

        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        labels = dbscan.fit_predict(df)
        labels_set = set(labels)
        num_of_clusters = len(labels_set)

        if -1 in labels_set:
            num_of_clusters -= 1

        if (num_of_clusters < 2) or (num_of_clusters > 25):
            scores.append(-20)
            all_labels.append('Poor')
            print('at Iteration: ', i, 'eps = ', eps, 'min_samples = ', min_samples, 'number of clusters = ',
                  num_of_clusters, 'continuing...')
            continue

        scores.append(shs(df, labels))
        all_labels.append(labels)
        print('at Iteration: ', i, 'eps = ', eps, 'min_samples = ', min_samples, 'score:', scores[-1],
              'number of clusters:', num_of_clusters)
        i += 1

        best_index = np.argmax(scores)
        best_parameters = combinations[best_index]
        best_labels = all_labels[best_index]
        best_score = scores[best_index]

        return {
            'best_epsilon': best_parameters[0],
            'best_min_samples': best_parameters[1],
            'best_labels': best_labels,
            'best_score': best_score
        }

# def GridSearch(combinations, df):
#
#     i = 0
#
#     for (eps, min_samples) in enumerate(combinations):
#         print('Epsilon:', eps)
#         print('Min samples:', min_samples)
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         labels = dbscan.fit_predict(df)
#         labels_set = set(labels)
#         num_of_clusters = len(labels_set)
#
#         if -1 in labels_set:
#             num_of_clusters -= 1
#
#         if (num_of_clusters < 2) or (num_of_clusters > 25):
#             scores.append(-20)
#             all_labels.append('Poor')
#             print('at Iteration: ', i, 'eps = ', eps, 'min_samples = ', min_samples, 'number of clusters = ', num_of_clusters, 'continuing...')
#             continue
#
#         scores.append(shs(df, labels))
#         all_labels.append(labels)
#         print('at Iteration: ', i, 'eps = ', eps, 'min_samples = ', min_samples,  'score:', scores[-1], 'number of clusters:', num_of_clusters)
#         i += 1
#
#     best_index = np.argmax(scores)
#     best_parameters = combinations[best_index]
#     best_labels = all_labels[best_index]
#     best_score = scores[best_index]
#
#     return {
#         'best_epsilon': best_parameters[0],
#         'best_min_samples': best_parameters[1],
#         'best_labels': best_labels,
#         'best_score': best_score
#     }

