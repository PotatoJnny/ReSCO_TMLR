"""Experiment config for ertest dataset."""
from ml_collections import config_dict


def get_config():
  """Get config for er benchmark graphs."""
  exp_config = dict(
      experiment=dict(
          batch_size=32,
          #batch_size = 1,
          t_schedule='exp_decay',
          #chain_length = 200000,
          chain_length=10000,
          #log_every_steps=1,
          #save_every_steps=100,
          init_temperature=1,
          decay_rate=0.001,
          final_temperature=0.001,
          save_root='',
      )
  )
  return config_dict.ConfigDict(exp_config)
