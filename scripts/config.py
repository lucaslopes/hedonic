experiment_params = {
  'samples': 10,
  'number_of_communities': 2,
  'community_size': 100,
  'probabilities': [1, .5, .25, .1, .05, .025, .01],
  'output_results_path': '~/Databases/hedonic',
}

methods = {
  'community_multilevel' : {
    'weights': None, # name of an edge attribute or a list containing edge weights.
    'return_levels': False, # if True, returns the multilevel result. If False, only the best level (corresponding to the best modularity) is returned.
    'resolution': 1, # the resolution parameter to use in the modularity measure. Smaller values result in a smaller number of larger clusters, while higher values yield a large number of small clusters. The classical modularity measure assumes a resolution parameter of 1.
  },
  'community_leiden' : {
    'objective_function': "CPM",
    'weights': None, # edge weights to be used. Can be a sequence or iterable or even an edge attribute name.
    'node_weights': None, # the node weights used in the Leiden algorithm.
    'resolution': 1, # the resolution parameter to use. Higher resolutions lead to more smaller communities, while lower resolutions lead to fewer larger communities.
    # 'normalize_resolution': False, # if set to true, the resolution parameter will be divided by the sum of the node weights. If this is not supplied, it will default to the node degree, or weighted degree in case edge_weights are supplied.
    'beta': 0.01, # 	parameter affecting the randomness in the Leiden algorithm. This affects only the refinement step of the algorithm.
    'initial_membership': None, # if provided, the Leiden algorithm will try to improve this provided membership. If no argument is provided, the aglorithm simply starts from the singleton partition.
    'n_iterations': 100, # the number of iterations to iterate the Leiden algorithm. Each iteration may improve the partition further. You can also set this parameter to a negative number, which means that the algorithm will be iterated until an iteration does not change the current membership vector any more.
  },
  'community_hedonic' : {
    'resolution': 1,
    'max_communities': 2,
  }
}