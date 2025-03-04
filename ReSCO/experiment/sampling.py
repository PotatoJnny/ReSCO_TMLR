"""Main class that runs sampler on the model to generate chains."""
import functools
import time
from ReSCO.common import math_util as math
from ReSCO.common import utils
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm


class Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config.experiment
    self.config_model = config.model
    self.parallel = False
    self.sample_idx = None
    self.num_saved_samples = config.get('nun_saved_samples', 4)
    if jax.local_device_count() != 1 and self.config.run_parallel:
      self.parallel = True

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    """Initializes model params, sampler state and gets the initial samples."""

    if self.config.evaluator == 'co_eval':
      sampler_init_state_fn = jax.vmap(sampler.make_init_state)
    else:
      sampler_init_state_fn = sampler.make_init_state
    model_init_params_fn = model.make_init_params
    rng_param, rng_x0, rng_state = jax.random.split(rnd, num=3)
    # params of the model
    params = model_init_params_fn(
        jax.random.split(rng_param, self.config.num_models)
    )
    # initial samples
    num_samples = self.config.batch_size * self.config.num_models
    x0 = model.get_init_samples(rng_x0, num_samples)
    # initial state of sampler
    state = sampler_init_state_fn(
        jax.random.split(rng_state, self.config.num_models)
    )
    return params, x0, state

  def _prepare_data(self, params, x, state):
    use_put_replicated = False
    reshape_all = True
    if self.config.evaluator != 'co_eval':
      if self.parallel:
        assert self.config.batch_size % jax.local_device_count() == 0
        mini_batch = self.config.batch_size // jax.local_device_count()
        bshape = (jax.local_device_count(),)
        x_shape = bshape + (mini_batch,) + self.config_model.shape
        use_put_replicated = True
        if self.sample_idx:
          self.sample_idx = jnp.array(
              [self.sample_idx]
              * (jax.local_device_count() // self.config.num_models)
          )
      else:
        reshape_all = False
        bshape = ()
        x_shape = (self.config.batch_size,) + self.config_model.shape
    else:
      if self.parallel:
        if self.config.num_models >= jax.local_device_count():
          assert self.config.num_models % jax.local_device_count() == 0
          num_models_per_device = (
              self.config.num_models // jax.local_device_count()
          )
          bshape = (jax.local_device_count(), num_models_per_device)
          x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
        else:
          assert self.config.batch_size % jax.local_device_count() == 0
          batch_size_per_device = (
              self.config.batch_size // jax.local_device_count()
          )
          use_put_replicated = True
          bshape = (jax.local_device_count(), self.config.num_models)
          x_shape = bshape + (batch_size_per_device,) + self.config_model.shape
          if self.sample_idx:
            self.sample_idx = jnp.array(
                [self.sample_idx]
                * (jax.local_device_count() // self.config.num_models)
            )
      else:
        bshape = (self.config.num_models,)
        x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
    fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    if reshape_all:
      if not use_put_replicated:
        state = jax.tree_map(fn_breshape, state)
        params = jax.tree_map(fn_breshape, params)
      else:
        params = jax.device_put_replicated(params, jax.local_devices())
        state = jax.device_put_replicated(state, jax.local_devices())
    x = jnp.reshape(x, x_shape)

    print('x shape: ', x.shape)
    print('state shape: ', state['steps'].shape)
    return params, x, state, bshape

  def _compile_sampler_step(self, step_fn):
    if not self.parallel:
      compiled_step = jax.jit(step_fn)
    else:
      compiled_step = jax.pmap(step_fn)
    return compiled_step

  def _compile_evaluator(self, obj_fn):
    if not self.parallel:
      compiled_obj_fn = jax.jit(obj_fn)
    else:
      compiled_obj_fn = jax.pmap(obj_fn)
    return compiled_obj_fn

  def _compile_fns(self, sampler, model, evaluator):
    if self.config.evaluator == 'co_eval':
      step_fn = jax.vmap(functools.partial(sampler.step, model=model))
      obj_fn = self._vmap_evaluator(evaluator, model)
    else:
      step_fn = functools.partial(sampler.step, model=model)
      obj_fn = evaluator.evaluate

    get_hop = jax.jit(self._get_hop)
    compiled_step = self._compile_sampler_step(step_fn)
    compiled_step_burnin = compiled_step
    compiled_step_mixing = compiled_step
    compiled_obj_fn = self._compile_evaluator(obj_fn)
    model_frwrd = jax.jit(model.forward)
    return (
        compiled_step_burnin,
        compiled_step_mixing,
        get_hop,
        compiled_obj_fn,
        model_frwrd,
    )

  def _get_hop(self, x, new_x):
    return (
        jnp.sum(abs(x - new_x))
        / self.config.batch_size
        / self.config.num_models
    )

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    raise NotImplementedError

  def vmap_evaluator(self, evaluator, model):
    raise NotImplementedError

  def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
    rnd = jax.random.PRNGKey(rnd_key)
    params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
    if params is None:
      print('Params is NONE')
      return False
    params, x, state, breshape = self._prepare_data(params, x, state)
    compiled_fns = self._compile_fns(sampler, model, evaluator)
    return [
        compiled_fns,
        state,
        params,
        rnd,
        x,
        saver,
        evaluator,
        breshape,
        model,
    ]

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, saver, rnd_key=0
  ):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    preprocessed_info = self.preprocess(
        model, sampler, evaluator, saver, rnd_key=0
    )
    if not preprocessed_info:
      return False
    self._compute_chain(*preprocessed_info)
    return True

  def get_results(self, model, sampler, evaluator, saver):
    self._get_chains_and_evaluations(model, sampler, evaluator, saver)



class CO_Experiment(Experiment):
  """Class used to run annealing schedule for CO problems."""

  def get_results(self, model, sampler, evaluator, saver):
    while True:
      if not self._get_chains_and_evaluations(model, sampler, evaluator, saver):
        break

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    data_list, x0, state = super()._initialize_model_and_sampler(
        rnd, model, sampler
    )
    if data_list is None:
      return None, x0, state
    sample_idx, params, reference_obj = zip(*data_list)
    params = flax.core.frozen_dict.unfreeze(utils.tree_stack(params))
    self.ref_obj = jnp.array(reference_obj)
    if self.config_model.name == 'mis':
      self.ref_obj = jnp.ones_like(self.ref_obj)
    self.sample_idx = jnp.array(sample_idx)
    return params, x0, state

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = jax.vmap(functools.partial(evaluator.evaluate, model=model))
    return obj_fn

  def _build_temperature_schedule(self, config):
    """Temperature schedule."""

    if config.t_schedule == 'constant':
      schedule = lambda step: step * 0 + config.init_temperature
    elif config.t_schedule == 'linear':
      schedule = optax.linear_schedule(
          config.init_temperature, config.final_temperature, config.chain_length
      )
    elif config.t_schedule == 'exp_decay':
      schedule = optax.exponential_decay(
          config.init_temperature,
          config.chain_length,
          config.decay_rate,
          end_value=config.final_temperature,
      )
    else:
      raise ValueError('Unknown schedule %s' % config.t_schedule)
    return schedule


  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    """Generates the chain of samples."""

    (
        chain,
        acc_ratios,
        hops,
        running_time,
        best_ratio,
        init_temperature,
        t_schedule,
        sample_mask,
        best_samples,
    ) = self._initialize_chain_vars(bshape)


    stp_burnin, _ , _, obj_fn, _ = compiled_fns
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    best_eval_val = jnp.ones((self.config.num_models,self.config.batch_size)) * -jnp.inf
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    value_chain = jnp.zeros((100, self.config.num_models, self.config.batch_size))
    reheat = True



    if reheat:
      shape = (self.config.num_models, self.config.batch_size)
      fake_step = jnp.ones(shape, dtype=jnp.int32)
      max_specific_heat = jnp.zeros(shape, dtype=jnp.float32)
      reheat_step = jnp.zeros(shape, dtype=jnp.int32)
      print_specific_heat = False
      skip_step = 200000
      wandering_length = 1000
      threshold = 0.5
      reheat_time = jnp.zeros((self.config.num_models,self.config.batch_size))
      trapped_num = jnp.zeros(shape, dtype=jnp.int32)
      trapped_threshold_length = jnp.ones(shape, dtype=jnp.int32) * wandering_length
      old_value = jnp.zeros(shape, dtype=jnp.float32)




    temp_shape = bshape+(self.config.batch_size,)
    init_temperature = jnp.ones(temp_shape, dtype=jnp.float32)
    params['temperature'] = t_schedule(0) * init_temperature

    for step in tqdm.tqdm(range(1, burn_in_length)):
        if reheat:
          temp = t_schedule(fake_step)
          temp = jnp.array(temp).reshape(params['temperature'].shape)
          params['temperature'] = temp
        else:
          temp = t_schedule(step)
          params['temperature'] = init_temperature * temp
        rng = jax.random.fold_in(rng, step)
        step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
        new_x, state, acc = stp_burnin(
            rng=step_rng,
            x=x,
            model_param=params,
            state=state,
            x_mask=params['mask'],
        )


        eval_val = obj_fn(samples=new_x, params=params) 
        eval_val = eval_val.reshape(self.config.num_models, -1)
        is_better = eval_val > best_eval_val 
        best_eval_val = jnp.maximum(eval_val, best_eval_val)
        value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
        sample_mask = sample_mask.reshape(best_eval_val.shape)

        br = np.array(best_eval_val[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)



        if self.config.save_samples or self.config_model.name == 'normcut':
          ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
          is_better = ratio > best_ratio
          best_ratio = jnp.maximum(ratio, best_ratio)
          sample_mask = sample_mask.reshape(best_ratio.shape)
          step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
          rnew_x = jnp.reshape(
            new_x,
            (self.config.num_models, self.config.batch_size)
            + self.config_model.shape,
          )
          chosen_samples = jnp.take_along_axis(
            rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
          )
          chosen_samples = jnp.squeeze(chosen_samples, -2)
          best_samples = jnp.where(
            jnp.expand_dims(is_better, -1), chosen_samples, best_samples
          )
          br = np.array(best_ratio[sample_mask])
          br = jax.device_put(br, jax.devices('cpu')[0])
          chain.append(br)



        if reheat:
          value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
          append_temp = temp.reshape(self.config.num_models, self.config.batch_size)
          value_diff = jnp.abs(eval_val - old_value)
          trapped_num = jnp.where(jnp.abs(value_diff) < threshold, trapped_num + jnp.ones_like(trapped_num),
                                  jnp.zeros_like(trapped_num))
          old_value = eval_val
          reheat_time_array = jnp.where(trapped_num >= trapped_threshold_length, jnp.ones_like(trapped_num),
                                        jnp.zeros_like(trapped_num))
          reheat_time = reheat_time + reheat_time_array
          if step >= skip_step:
            specific_heat = (jnp.var(value_chain, axis=0) / (append_temp ** 2))
            specific_heat = jnp.where(reheat_time == jnp.zeros_like(reheat_time), specific_heat, jnp.zeros_like(reheat_time))
            if print_specific_heat:
              print('specific_heat', jnp.mean(specific_heat))
            max_specific_heat = jnp.maximum(specific_heat, max_specific_heat)
            reheat_step = jnp.where(specific_heat >= max_specific_heat, fake_step, reheat_step)
            fake_step = jnp.where(trapped_num >= trapped_threshold_length, reheat_step - jnp.ones_like(reheat_step), fake_step)


        x = new_x
        if reheat:
          fake_step = fake_step + jnp.ones_like(fake_step)


    running_time = 0
    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      if reheat:
        temp = t_schedule(fake_step)
        temp = jnp.array(temp).reshape(params['temperature'].shape)
      else:
        temp = t_schedule(step)
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
          x_mask=params['mask'],
      )
      running_time += time.time() - start


      eval_val = obj_fn(samples=new_x, params=params)
      eval_val = eval_val.reshape(self.config.num_models, -1)
      is_better = eval_val > best_eval_val
      best_eval_val = jnp.maximum(eval_val, best_eval_val)
      sample_mask = sample_mask.reshape(best_eval_val.shape)
      br = np.array(best_eval_val[sample_mask])
      br = jax.device_put(br, jax.devices('cpu')[0])
      chain.append(br)


      if self.config.save_samples or self.config_model.name == 'normcut':
        ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
        is_better = ratio > best_ratio
        best_ratio = jnp.maximum(ratio, best_ratio)
        sample_mask = sample_mask.reshape(best_ratio.shape)
        step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
        rnew_x = jnp.reshape(
          new_x,
          (self.config.num_models, self.config.batch_size)
          + self.config_model.shape,
        )
        chosen_samples = jnp.take_along_axis(
          rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
        )
        chosen_samples = jnp.squeeze(chosen_samples, -2)
        best_samples = jnp.where(
          jnp.expand_dims(is_better, -1), chosen_samples, best_samples
        )
        br = np.array(best_ratio[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)       


      if reheat:
        # we don't calculate specific heat for the mixing phase, since we don't update the critical temperature anymore in case it becomes too small
        value_diff = jnp.abs(eval_val - old_value)
        trapped_num = jnp.where(jnp.abs(value_diff) < threshold, trapped_num + jnp.ones_like(trapped_num),
                                jnp.zeros_like(trapped_num))
        old_value = eval_val
        reheat_time_array = jnp.where(trapped_num >= trapped_threshold_length, jnp.ones_like(trapped_num),
                                      jnp.zeros_like(trapped_num))
        reheat_time = reheat_time + reheat_time_array
        fake_step = jnp.where(trapped_num >= trapped_threshold_length, reheat_step - jnp.ones_like(reheat_step), fake_step)



      x = new_x
      if reheat:
        fake_step = fake_step + jnp.ones_like(fake_step)


    best_value = jnp.max(best_eval_val, axis=-1).reshape(-1)
    sample_mask = sample_mask.reshape(best_value.shape)
    best_value = best_value[sample_mask]
    best_ratio = best_value / self.ref_obj[sample_mask]
    if not (self.config.save_samples or self.config_model.name == 'normcut'):
      best_samples = []
    saver.save_co_resuts(
        chain, best_value[sample_mask], best_ratio[sample_mask], running_time, best_samples
    )








  def _initialize_chain_vars(self, bshape):
    t_schedule = self._build_temperature_schedule(self.config)
    sample_mask = self.sample_idx >= 0
    chain = []
    acc_ratios = []
    hops = []
    running_time = 0
    best_ratio = jnp.ones(self.config.num_models, dtype=jnp.float32) * -1e9
    init_temperature = jnp.ones(bshape, dtype=jnp.float32)
    dim = math.prod(self.config_model.shape)
    best_samples = jnp.zeros([self.config.num_models, dim])
    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        best_ratio,
        init_temperature,
        t_schedule,
        sample_mask,
        best_samples,
    )

