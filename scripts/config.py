methods = {
  'community_multilevel' : {
    'weights': None, # name of an edge attribute or a list containing edge weights
    'return_levels': False, # if True, returns the multilevel result. If False, only the best level (corresponding to the best modularity) is returned
    'resolution': 1, # the resolution parameter to use in the modularity measure. Smaller values result in a smaller number of larger clusters, while higher values yield a large number of small clusters. The classical modularity measure assumes a resolution parameter of 1
  },
  'community_leiden' : {
    'objective_function': "CPM",
    'weights': None, # edge weights to be used. Can be a sequence or iterable or even an edge attribute name
    'resolution': 1, # the resolution parameter to use. Higher resolutions lead to more smaller communities, while lower resolutions lead to fewer larger communities
    'beta': 0.01, # parameter affecting the randomness in the Leiden algorithm. This affects only the refinement step of the algorithm
    'initial_membership': None, # if provided, the Leiden algorithm will try to improve this provided membership. If no argument is provided, the aglorithm simply starts from the singleton partition
    'n_iterations': 2, # the number of iterations to iterate the Leiden algorithm. Each iteration may improve the partition further
    'node_weights': None, # the node weights used in the Leiden algorithm
  },
  'community_hedonic' : {
    'resolution': 1,
    'max_communities': 2,
  }
}


