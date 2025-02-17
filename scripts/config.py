# experiment_params['output_results_path'] = f"~/Databases/hedonic/{experiment_params['number_of_communities']}C_{experiment_params['community_size']}N_{len(methods)}_methods_{len(experiment_params['probabilities'])}_probabilities_{len(experiment_params['difficulties'])}_difficulties"
# experiment_params['output_results_path'] = f"~/Databases/hedonic/PHYSA_{experiment_params['community_size']*experiment_params['number_of_communities']}"
experiment_params = {
  'output_results_path': '/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V2040',
  'number_of_communities': 6,
  'community_size': int(2040/6),
  'network_seeds': [i for i in range(100)],
  'partition_seeds': [i for i in range(100)],
  # 'noises': [0.01, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1],  # Synthetic networks
  'noises': [.01, .025, .05, .075, .1, .15, .2, .25, .3, .35, .4, .45, .5],  # Real networks
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
  'Mirror': {
    'method_call_name': 'community_mirror',
    'parameters': {
      'initial_membership': None,
    },
  },
  'OnePass': {
    'method_call_name': 'community_onepass_improvement',
    'parameters': {
      'initial_membership': None,
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
  # 'Leiden': {
  #   'method_call_name': 'community_leiden',
  #   'parameters': {
  #     'objective_function': "CPM",
  #     'weights': None,
  #     'node_weights': None,
  #     'resolution': 1,
  #     'beta': 0.01,
  #     'initial_membership': None,
  #     'n_iterations': -1,
  #     'can_create_new_clusters': False,
  #   },
  # },
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
      'only_first_phase': True,
      'can_create_new_clusters': False,
    },
  },
  # 'Hedonic Games': { # This method was replaced by the method above (Leiden with hedonic)
  #   'method_call_name': 'community_hedonic',
  #   'parameters': {
  #     'resolution': 1,
  #     'initial_membership': None,
  #   },
  # },
  # 'Louvain': {
  #   'method_call_name': 'community_multilevel',
  #   'parameters': {
  #     'weights': None,
  #     'return_levels': False,
  #     'resolution': 1,
  #   },
  # },
}