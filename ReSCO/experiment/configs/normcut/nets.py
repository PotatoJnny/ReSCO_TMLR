"""Experiment config for nets dataset."""

from ml_collections import config_dict


def get_config():
  """Get config for er benchmark graphs."""
  exp_config = dict(
      experiment=dict(
          batch_size=1,
          t_schedule='exp_decay',
          #chain_length = 80000,
          chain_length=800000,
          log_every_steps=1,
          save_every_steps=100,
          init_temperature=2,
          decay_rate=0.01,
          final_temperature=0.0000001,
          save_root='',
      )
  )

  return config_dict.ConfigDict(exp_config)
