"""Experiment config for maxclique rb dataset."""
from ml_collections import config_dict


def get_config():
  """Get config."""
  exp_config = dict(
      experiment=dict(
          batch_size=1,
          t_schedule='exp_decay',
          chain_length=10001,
          log_every_steps=1,
          init_temperature=1.0,
          decay_rate=0.001,
          final_temperature=0.001,
          save_root='',
      )
  )
  return config_dict.ConfigDict(exp_config)
