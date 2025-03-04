"""Config for rb job."""

from ml_collections import config_dict

def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='fhmm',
          sampler='path_auxiliary',
          sweep=[
              {
                  'model_config.num_categories': [2],
                  'model_config.shape': ['(200, 50)', '(1000, 10)'],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'model_config.num_categories': [2],
                  'model_config.shape': ['(200, 50)', '(1000, 10)'],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'model_config.num_categories': [2],
                  'model_config.shape': ['(200, 50)', '(1000, 10)'],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
