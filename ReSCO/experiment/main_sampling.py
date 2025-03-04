"""Main script for sampling based experiments."""
import importlib
from absl import app
from absl import flags
from ReSCO.common import configs as common_configs
from ReSCO.common import utils
import ReSCO.common.experiment_saver as saver_mod
from ml_collections import config_flags


FLAGS = flags.FLAGS
_EXPERIMENT_CONFIG = config_flags.DEFINE_config_file(
    'config', './ReSCO/common/configs.py'
)
_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')
_RUN_LOCAL = flags.DEFINE_boolean('run_local', False, 'if runnng local')


def get_save_dir(config):
  if _RUN_LOCAL.value:
    save_folder = config.model.get('save_dir_name', config.model.name)
    save_root = './ReSCO/results/' + save_folder
  else:
    save_root = config.experiment.save_root
  return save_root


def get_main_config():
  """Merge experiment, model and sampler config."""
  config = common_configs.get_config()
  if (
      'graph_type' not in _MODEL_CONFIG.value
  ):
    config.update(_EXPERIMENT_CONFIG.value)
  config.sampler.update(_SAMPLER_CONFIG.value)
  config.model.update(_MODEL_CONFIG.value)
  if config.model.get('graph_type', None):
    graph_config = importlib.import_module(
        'ReSCO.models.configs.%s.%s'
        % (config.model['name'], config.model['graph_type'])
    )
    config.model.update(graph_config.get_model_config(config.model['cfg_str']))
    co_exp_default_config = importlib.import_module(
        'ReSCO.experiment.configs.co_experiment'
    )
    config.experiment.update(co_exp_default_config.get_co_default_config())
    config.update(_EXPERIMENT_CONFIG.value)
    config.experiment.num_models = config.model.num_models


  return config


def main(_):
  config = get_main_config()
  utils.setup_logging(config)

  # model
  model_mod = importlib.import_module('ReSCO.models.%s' % config.model.name)
  model = model_mod.build_model(config)

  # sampler
  sampler_mod = importlib.import_module(
      'ReSCO.samplers.%s' % config.sampler.name
  )
  sampler = sampler_mod.build_sampler(config)

  # experiment
  experiment_mod = getattr(
      importlib.import_module('ReSCO.experiment.sampling'),
      f'{config.experiment.name}',
  )
  experiment = experiment_mod(config)

  # evaluator
  evaluator_mod = importlib.import_module(
      'ReSCO.evaluators.%s' % config.experiment.evaluator
  )
  evaluator = evaluator_mod.build_evaluator(config)

  # saver
  saver = saver_mod.build_saver(get_save_dir(config), config)

  # chain generation
  experiment.get_results(model, sampler, evaluator, saver)


if __name__ == '__main__':
  app.run(main)
