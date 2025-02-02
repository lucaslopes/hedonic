experiment_params = {
  'output_results_path': '~/Databases/hedonic/',
  'number_of_communities': 2,
  'community_size': 1000,
  'network_seeds': [i for i in range(100)],
  'partition_seeds': [i for i in range(100)],
  'noises': [0.01, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1],
  'probabilities': [.10, .09, .08, .07, .06, .05, .04, .03, .02, .01],
  'difficulties': [.75, .7, .65, .6, .55, .5, .4, .3, .2, .1],
}

methods = {
  'GroundTruth': {
    'method_call_name': 'community_groundtruth',
    'parameters': {
      'groundtruth': None,
    },
  },
  'Spectral': {
    'method_call_name': 'community_leading_eigenvector',
    'parameters': {
      'clusters': None,
      'weights': None,
      'arpack_options': None,
    },
  },
  'Louvain': {
    'method_call_name': 'community_multilevel',
    'parameters': {
      'weights': None,
      'return_levels': False,
      'resolution': 1,
    },
  },
  'Leiden': {
    'method_call_name': 'community_leiden',
    'parameters': {
      'objective_function': "CPM",
      'weights': None,
      'node_weights': None,
      'resolution': 1,
      'beta': 0.01,
      'initial_membership': None,
      'n_iterations': -1,
    },
  },
  'Hedonic': {
    'method_call_name': 'community_leiden',
    'parameters': {
      'objective_function': "CPM",
      'weights': None,
      'node_weights': None,
      'resolution': 1,
      'beta': 0.01,
      'initial_membership': None,
      'n_iterations': -1,
      'hedonic': True,
    },
  },
  # 'Hedonic Games': { # This method was replaced by the method above (Leiden with hedonic)
  #   'method_call_name': 'community_hedonic',
  #   'parameters': {
  #     'resolution': 1,
  #     'initial_membership': None,
  #   },
  # },
  'OnePass': {
    'method_call_name': 'community_onepass_improvement',
    'parameters': {
      'initial_membership': None,
    },
  },
  'Mirror': {
    'method_call_name': 'community_mirror',
    'parameters': {
      'initial_membership': None,
    },
  },
}

# experiment_params['output_results_path'] = f"~/Databases/hedonic/{experiment_params['number_of_communities']}C_{experiment_params['community_size']}N_{len(methods)}_methods_{len(experiment_params['probabilities'])}_probabilities_{len(experiment_params['difficulties'])}_difficulties"
experiment_params['output_results_path'] = f"~/Databases/hedonic/PHYSA_{experiment_params['community_size']*experiment_params['number_of_communities']}"