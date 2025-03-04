"""Main class that runs sampler on the model to generate chains."""
import functools
import time
from discs.common import math_util as math
from discs.common import utils
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


class Sampling_Experiment(Experiment):
  """Class used to run classical graphical models and computes ESS."""

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = evaluator.evaluate
    return obj_fn

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
    assert self.config.num_models == 1
    (chain, acc_ratios, hops, running_time, samples) = (
        self._initialize_chain_vars()
    )
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    # sample used to map for ess computation
    rng_x0_ess, rng = jax.random.split(rng)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    stp_burnin, stp_mixing, get_hop, obj_fn, _ = compiled_fns
    get_mapped_samples, eval_metric = self._compile_additional_fns(evaluator)
    rng = jax.random.PRNGKey(10)
    selected_chains = jax.random.choice(
        rng,
        jnp.arange(self.config.batch_size),
        shape=(self.num_saved_samples,),
        replace=False,
    )

    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    for step in tqdm.tqdm(range(1, burn_in_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      running_time += time.time() - start

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      mapped_sample = get_mapped_samples(new_x, x0_ess)
      mapped_sample = jax.device_put(mapped_sample, jax.devices('cpu')[0])
      chain.append(mapped_sample)
      x = new_x

    chain = jnp.array(chain)
    if self.parallel:
      chain = jnp.array([chain])
      rng = jnp.array([rng])
      num_ll_calls = int(state['num_ll_calls'][0])
    else:
      num_ll_calls = int(state['num_ll_calls'])
    ess = obj_fn(samples=chain, rnd=rng)
    metrics = eval_metric(ess, running_time, num_ll_calls)
    saver.save_results(acc_ratios, hops, metrics, running_time)
    if self.config.save_samples or self.config.get_estimation_error:
      if self.config.save_samples and self.config_model.name in [
          'rbm',
          'resnet',
      ]:
        saver.dump_samples(samples, visualize=False)
      elif (
          self.config.get_estimation_error
          and self.config_model.name == 'bernoulli'
      ):
        saver.dump_samples(samples, visualize=False)
        # samples= np.array(samples)
        params = params['params'][0].reshape(self.config_model.shape)
        saver.dump_params(params)
        # saver.plot_estimation_error(model, params, samples)

  def _initialize_chain_vars(self):
    chain = []
    acc_ratios = []
    hops = []
    samples = []
    running_time = 0

    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        samples,
    )

  def _compile_additional_fns(self, evaluator):
    get_mapped_samples = jax.jit(self._get_mapped_samples)
    eval_metric = jax.jit(evaluator.get_eval_metrics)
    return get_mapped_samples, eval_metric

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((-1,) + self.config_model.shape)
    samples = samples.reshape(samples.shape[0], -1)
    x0_ess = x0_ess.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)


class Text_Infilling_Experiment(Sampling_Experiment):
  """Class used to sample sentences for text infilling."""

  def get_results(self, model, sampler, evaluator, saver):
    obj_fn = jax.jit(evaluator.evaluate)
    infill_sents = []
    infill_sents_topk = []
    rnd_key = 0
    while True:
      contin, sents, sents_topk = self._get_chains_and_evaluations(
          model, sampler, evaluator, saver, rnd_key=rnd_key
      )
      rnd_key += 1
      if not contin:
        break
      infill_sents.extend(sents)
      if self.config.use_topk:
        infill_sents_topk.extend(sents_topk)
    res = obj_fn(infill_sents, self.config_model.data_root)
    if self.config.use_topk:
      res_topk = evaluator.evaluate(
          infill_sents_topk, self.config_model.data_root
      )
    else:
      res_topk = []
    saver.save_lm_results(res, res_topk)

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, saver, rnd_key=0
  ):
    preprocessed_info = self.preprocess(
        model, sampler, evaluator, saver, rnd_key=0
    )
    if not preprocessed_info:
      return False, None, None
    sentences = []
    loglikes = []
    topk_sentences = []

    obj_fn = preprocessed_info[0][-1]
    for i in range(self.config.num_same_resample):
      sent, rng, loglike = self._compute_chain(*preprocessed_info)
      if self.config.use_topk:
        sent = str(i) + ' ' + sent
        loglikes.append(loglike)
      sentences.append(sent)
      preprocessed_info[3] = rng

    if self.config.use_topk:
      sent_to_loglike = dict(zip(sentences, loglikes))
      sorted_sent = {
          k: v
          for k, v in sorted(sent_to_loglike.items(), key=lambda item: item[1])
      }
      topk_sentences = list(sorted_sent.keys())[-self.config.topk_num :]
      for i, sent in enumerate(topk_sentences):
        topk_sentences[i] = sent[2:]
      for i, sent in enumerate(sentences):
        sentences[i] = sent[2:]

    return True, sentences, topk_sentences

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
    assert self.config.num_models == 1
    (_, acc_ratios, hops, running_time, _) = self._initialize_chain_vars()

    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    stp_burnin, stp_mixing, get_hop, _, model_frwrd = compiled_fns

    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    for step in tqdm.tqdm(range(1, burn_in_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      running_time += time.time() - start
      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    loglike = 0
    if self.config.use_topk:
      x = x.astype(jnp.float32)
      loglike = model_frwrd(params, x)[0]

    sampled_sentence = model.decode(x, params)
    print('Sampled Sentence: ', sampled_sentence, 'Likelihood: ', loglike)
    return sampled_sentence, rng, loglike


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


    stp_burnin, stp_mixing, get_hop, obj_fn, _ = compiled_fns
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    best_eval_val = jnp.ones((self.config.num_models,self.config.batch_size)) * -jnp.inf
    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    #temp_chain = jnp.zeros((self.config.chain_length,self.config.num_models,self.config.batch_size))
    #value_chain = jnp.zeros((self.config.chain_length,self.config.num_models,self.config.batch_size))
    value_chain = jnp.zeros((100, self.config.num_models, self.config.batch_size))
    #best_value_chain = jnp.zeros((self.config.chain_length,self.config.num_models,self.config.batch_size))

    #record_value_chain = jnp.zeros((int(self.config.chain_length / self.config.log_every_steps), self.config.num_models, self.config.batch_size))
    #record_specific_heat_chain = jnp.zeros((self.config.chain_length, self.config.num_models, self.config.batch_size))
    #record_specific_heat_differece_chain = jnp.zeros((self.config.chain_length, self.config.num_models, self.config.batch_size))

    record_value_chain = []
    record_temp_chain = []

    print_specific_heat = False
    reheat = True
    skip_step =200000
    wandering_length = 1000
    #reheat_type = 'Half-then-half'
    reheat_from_best = False
    start_new_chain = False
    #reheat_type = 'Best_Temp'
    #reheat_type = 'Initial_Temp'
    reheat_type = 'specific_heat'
    #reheat_type = 'other'
    reheat_time = jnp.zeros((self.config.num_models,self.config.batch_size))
    threshold = 0.5
    distance_array = []

    shape = (self.config.num_models, self.config.batch_size)
    fake_step = jnp.ones(shape, dtype=jnp.int32)
    if self.config_model.name == '_normcut':
      fake_step = jnp.ones((8,32), dtype=jnp.int32)
    trapped_num = jnp.zeros(shape, dtype=jnp.int32)
    trapped_threshold_length = jnp.ones(shape, dtype=jnp.int32) * wandering_length
    old_value = jnp.zeros(shape, dtype=jnp.float32)

    if reheat and reheat_type == 'Best_Temp':
      Best_Temp_Step = jnp.ones(shape, dtype=jnp.int32)

    if reheat and reheat_type == 'specific_heat':
      max_specific_heat = jnp.zeros(shape, dtype=jnp.float32)
      #specific_heat = jnp.zeros(shape, dtype=jnp.float32)
      reheat_step = jnp.zeros(shape, dtype=jnp.int32)
      #specific_heat_chain = jnp.zeros((self.config.chain_length,self.config.num_models,self.config.batch_size))

    if reheat and reheat_from_best:
      best_x = x



    temp_shape = bshape+(self.config.batch_size,)
    init_temperature = jnp.ones(temp_shape, dtype=jnp.float32)
    params['temperature'] = t_schedule(0) * init_temperature

    for step in tqdm.tqdm(range(1, burn_in_length)):
        if reheat:
          temp = t_schedule(fake_step)
          temp = jnp.array(temp).reshape(params['temperature'].shape)
          if self.config_model.name == '_normcut':
            params['temperature'] = temp[0]
          else:
            params['temperature'] = temp
        else:
          temp = t_schedule(step)
          if self.config_model.name == '_normcut':
            params['temperature'] = temp[0]
          else:
            params['temperature'] = init_temperature * temp
        #print(temp)
        rng = jax.random.fold_in(rng, step)
        step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
        new_x, state, acc = stp_burnin(
            rng=step_rng,
            x=x,
            model_param=params,
            state=state,
            x_mask=params['mask'],
        )

        if step % self.config.log_every_steps == 0:
          eval_val = obj_fn(samples=new_x, params=params) #shape (device_count, num_models/device_count, batch_size)
          eval_val = eval_val.reshape(self.config.num_models, -1)
          is_better = eval_val > best_eval_val # shape (num_models, batch_size)
          best_eval_val = jnp.maximum(eval_val, best_eval_val)
          #record_value_chain = record_value_chain.at[step // self.config.log_every_steps - 1].set(eval_val)
          #best_value_chain = best_value_chain.at[step - 1].set(best_eval_val)
          value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
          #if step >= 100:
            #specific_heat = (jnp.var(value_chain, axis=0) / (temp.reshape(self.config.num_models, self.config.batch_size) ** 2))
            #record_specific_heat_chain = record_specific_heat_chain.at[step - 1].set(specific_heat)
            #record_specific_heat_differece_chain = record_specific_heat_differece_chain.at[step - 1].set(jnp.abs(eval_val - old_value) / (temp.reshape(self.config.num_models, self.config.batch_size) - old_temperature.reshape(self.config.num_models, self.config.batch_size)))


          record_value_chain.append(eval_val[0][0])
          record_temp_chain.append(temp[0][0])  
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
            #print('best samples shape:',best_samples.shape )
            



          if reheat:
            value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
            append_temp = temp.reshape(self.config.num_models, self.config.batch_size)
           # temp_chain = temp_chain.at[step - 1].set(append_temp)
            value_diff = jnp.abs(eval_val - old_value)
            trapped_num = jnp.where(jnp.abs(value_diff) < threshold, trapped_num + jnp.ones_like(trapped_num),
                                    jnp.zeros_like(trapped_num))
            #print(jnp.mean(value_diff))
            #print('value diff',value_diff[0][0])
            #print('eval val',eval_val[0][0])
            old_value = eval_val
            reheat_time_array = jnp.where(trapped_num >= trapped_threshold_length, jnp.ones_like(trapped_num),
                                          jnp.zeros_like(trapped_num))
            reheat_time = reheat_time + reheat_time_array
            if reheat_from_best:
              x_is_better = is_better.reshape(bshape + (self.config.batch_size,))[:, :, :, None]
              best_x = jnp.where(x_is_better, new_x, best_x)
            if reheat_type == 'Half-then-half':
              half_fake_step = fake_step // 2
              fake_step = jnp.where(trapped_num >= trapped_threshold_length, half_fake_step, fake_step)
            if reheat_type == 'Best_Temp':
              Best_Temp_Step = jnp.where(is_better, fake_step, Best_Temp_Step)
              fake_step = jnp.where(trapped_num >= trapped_threshold_length, Best_Temp_Step - jnp.ones_like(Best_Temp_Step), fake_step)
            if reheat_type == 'specific_heat' and step >= skip_step:
              # calculate var(value[step-100 ~ step - 1])/ temp ** 2
              specific_heat = (jnp.var(value_chain, axis=0) / (append_temp ** 2))
              specific_heat = jnp.where(reheat_time == jnp.zeros_like(reheat_time), specific_heat, jnp.zeros_like(reheat_time))
              if print_specific_heat:
                print('specific_heat', jnp.mean(specific_heat))
              #print('reheat time', jnp.sum(reheat_time))
              #specific_heat_chain = specific_heat_chain.at[step - 1].set(specific_heat)
              max_specific_heat = jnp.maximum(specific_heat, max_specific_heat)
              reheat_step = jnp.where(specific_heat >= max_specific_heat, fake_step, reheat_step)
              fake_step = jnp.where(trapped_num >= trapped_threshold_length, reheat_step - jnp.ones_like(reheat_step), fake_step)
              #best_value = jnp.max(best_eval_val, axis=-1).reshape(-1)
              #sample_mask = sample_mask.reshape(best_value.shape)
              #best_value = best_value[sample_mask]
              #best_ratio = best_value / self.ref_obj[sample_mask]
              #print('mean_best_ratio', jnp.mean(best_ratio))
            if reheat_from_best:
              reheat_array_for_x = reheat_time_array.reshape(bshape + (self.config.batch_size,))[:, :, :, None]
              new_x = jnp.where(reheat_array_for_x, best_x, new_x)
            if start_new_chain:
              fake_step = jnp.where(trapped_num >= trapped_threshold_length, jnp.zeros_like(fake_step),
                                    fake_step)
              #reheat_array_for_x = reheat_time_array.reshape(bshape + (self.config.batch_size,))[:, :, :, None]
              reheat_array_for_x = reheat_time_array[:, :, None]
              new_x = jnp.where(reheat_array_for_x, jnp.array(np.random.randint(2, size=jnp.shape(new_x))), new_x)



        #if self.config.get_additional_metrics:
          # avg over all models
          #acc = jnp.mean(acc)
          #acc_ratios.append(acc)
          # hop avg over batch size and num models
          #hops.append(get_hop(x, new_x))

        x = new_x
        if reheat:
          fake_step = fake_step + jnp.ones_like(fake_step)
          old_temperature = temp


    running_time = 0
    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      if reheat:
        temp = t_schedule(fake_step)
        temp = jnp.array(temp).reshape(params['temperature'].shape)
        if self.config_model.name == '_normcut':
          params['temperature'] = temp[0]
        else:
          params['temperature'] = temp
      else:
        temp = t_schedule(step)
        if self.config_model.name == '_normcut':
          params['temperature'] = temp[0]
        else:
          params['temperature'] = temp * init_temperature
      #print(temp[0][0])
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

      if step % self.config.log_every_steps == 0:
        eval_val = obj_fn(samples=new_x, params=params)
        eval_val = eval_val.reshape(self.config.num_models, -1)
        is_better = eval_val > best_eval_val
        best_eval_val = jnp.maximum(eval_val, best_eval_val)
        #record_value_chain = record_value_chain.at[step // self.config.log_every_steps - 1].set(eval_val)

        #value_chain = value_chain.at[step - 1].set(eval_val)
        #best_value_chain = best_value_chain.at[step - 1].set(best_eval_val)


        #sample_mask = sample_mask.reshape(best_ratio.shape)
        #interval_improvement = best_ratio - old_best_ratio

        #value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
        #specific_heat = (jnp.var(value_chain, axis=0) / (temp.reshape(self.config.num_models, self.config.batch_size) ** 2))
        #record_specific_heat_chain = record_specific_heat_chain.at[step - 1].set(specific_heat)
        #record_specific_heat_differece_chain = record_specific_heat_differece_chain.at[step - 1].set(jnp.abs(eval_val - old_value) / (temp.reshape(self.config.num_models, self.config.batch_size) - old_temperature.reshape(self.config.num_models, self.config.batch_size)))

         


        record_value_chain.append(eval_val[0][0])
        record_temp_chain.append(temp[0][0])

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
          


        if reheat:
          #value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
          #append_temp = temp.reshape(self.config.num_models, self.config.batch_size)
          #temp_chain = temp_chain.at[step - 1].set(append_temp)
          value_diff = jnp.abs(eval_val - old_value)
          trapped_num = jnp.where(jnp.abs(value_diff) < threshold, trapped_num + jnp.ones_like(trapped_num),
                                  jnp.zeros_like(trapped_num))
          #print('value diff',value_diff[0][0])
          #print('eval val',eval_val[0][0])
          old_value = eval_val
          reheat_time_array = jnp.where(trapped_num >= trapped_threshold_length, jnp.ones_like(trapped_num),
                                        jnp.zeros_like(trapped_num))
          reheat_time = reheat_time + reheat_time_array
          if reheat_from_best:
            x_is_better = is_better.reshape(bshape + (self.config.batch_size,))[:, :, :, None]
            best_x = jnp.where(x_is_better, new_x, best_x)
          if reheat_type == 'Half-then-half':
            half_fake_step = fake_step // 2
            fake_step = jnp.where(trapped_num >= trapped_threshold_length, half_fake_step, fake_step)
          if reheat_type == 'Best_Temp':
            Best_Temp_Step = jnp.where(is_better, fake_step, Best_Temp_Step)
            fake_step = jnp.where(trapped_num >= trapped_threshold_length, Best_Temp_Step - jnp.ones_like(Best_Temp_Step), fake_step)
          if reheat_type == 'specific_heat':
            # calculate var(value[step-100 ~ step - 1])/ temp ** 2
            #specific_heat = (jnp.var(value_chain, axis=0) / (append_temp ** 2))
            #print('specific_heat', jnp.mean(specific_heat))
            #specific_heat_chain = specific_heat_chain.at[step - 1].set(specific_heat)
            #max_specific_heat = jnp.maximum(specific_heat, max_specific_heat)
            #reheat_step = jnp.where(specific_heat >= max_specific_heat, fake_step, reheat_step)
            fake_step = jnp.where(trapped_num >= trapped_threshold_length, reheat_step - jnp.ones_like(reheat_step), fake_step)
            #best_value = jnp.max(best_eval_val, axis=-1).reshape(-1)
            #sample_mask = sample_mask.reshape(best_value.shape)
            #best_value = best_value[sample_mask]
            #best_ratio = best_value / self.ref_obj[sample_mask]
            #print('mean_best_ratio', jnp.mean(best_ratio))
          if reheat_from_best:
            reheat_array_for_x = reheat_time_array.reshape(bshape + (self.config.batch_size,))[:, :, :, None]
            new_x = jnp.where(reheat_array_for_x, best_x, new_x)
          if start_new_chain:
            fake_step = jnp.where(trapped_num >= trapped_threshold_length, jnp.zeros_like(fake_step),
                                  fake_step)
            reheat_array_for_x = reheat_time_array[:,:,None]
            new_x = jnp.where(reheat_array_for_x, jnp.array(np.random.randint(2, size=jnp.shape(new_x))), new_x)





      #if self.config.get_additional_metrics:
        # avg over all models
        #acc = jnp.mean(acc)
        #acc_ratios.append(acc)
        # hop avg over batch size and num models
        #hops.append(get_hop(x, new_x))

      x = new_x
      if reheat:
        fake_step = fake_step + jnp.ones_like(fake_step)
        old_temperature = temp


    import csv
    import os
    import matplotlib.pyplot as plt

    folder_name = 'Record Running Results'
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)
    # Define the CSV filename
    csv_filename = 'record_value_chain.csv'

    # Full path to the CSV file
    csv_filepath = os.path.join(folder_name, csv_filename)

    # Write the list to a CSV file
    with open(csv_filepath, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # If you want each item on a separate row:
        for value in record_value_chain:
            writer.writerow([value])

     
    csv_filename = 'record_temp_chain.csv'

    # Full path to the CSV file
    csv_filepath = os.path.join(folder_name, csv_filename)

    # Write the list to a CSV file
    with open(csv_filepath, mode='w', newline='') as csv_file:
         writer = csv.writer(csv_file)
         # If you want each item on a separate row:
         for value in record_temp_chain:
            writer.writerow([value])
    

    best_value = jnp.max(best_eval_val, axis=-1).reshape(-1)
    std_value = jnp.std(best_eval_val, axis=-1).reshape(-1)
    sample_mask = sample_mask.reshape(best_value.shape)
    best_value = best_value[sample_mask]
    best_ratio = best_value / self.ref_obj[sample_mask]
    std_value = std_value[sample_mask]
    print('running time: ', running_time)
    print('mean best ratio: ', jnp.mean(best_ratio))
    print('mean best value: ', jnp.mean(best_value))
    print('std', jnp.mean(std_value))
    print('reheat time: ', jnp.sum(reheat_time))
    #raise ValueError('stop here')

    '''
    # save reheat_saver as a csv file, each list writing as a column
    import csv
    import os
    import matplotlib.pyplot as plt

    folder_name = 'Running Results'
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)

    # record value chain in a csv file
    #record_value_chain = np.array(record_value_chain.reshape(int(self.config.chain_length / self.config.log_every_steps), self.config.num_models*self.config.batch_size))
    #record_value_chain = list(record_value_chain)
    #with open(folder_name + '/record_value_chain.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    
    #    for rows in record_value_chain:
    #        writer.writerow(rows)

    record_specific_heat_chain = np.array(record_specific_heat_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size))
    record_specific_heat_chain = list(record_specific_heat_chain)
    with open(folder_name + '/record_specific_heat_chain.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rows in record_specific_heat_chain:
            writer.writerow(rows)

    record_specific_heat_differece_chain = np.array(record_specific_heat_differece_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size))
    record_specific_heat_differece_chain = list(record_specific_heat_differece_chain)
    with open(folder_name + '/record_specific_heat_differece_chain.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rows in record_specific_heat_differece_chain:
            writer.writerow(rows)

    record_value_chain = np.array(record_value_chain.reshape(int(self.config.chain_length / self.config.log_every_steps), self.config.num_models*self.config.batch_size))
    record_value_chain = list(record_value_chain)
    with open(folder_name + '/record_value_chain.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rows in record_value_chain:
            writer.writerow(rows)




    # record temperature chain and value chain in two csv files
    #temp_chain = temp_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size)
    #temp_chain = np.array(temp_chain)
    #temp_chain = list(temp_chain)
    #value_chain = value_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size)
    #value_chain = np.array(value_chain)
    #value_chain = list(value_chain)
    #best_value_chain = best_value_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size)
    #best_value_chain = np.array(best_value_chain)
    #best_value_chain = list(best_value_chain)

    # save temp_chain and value_chain as csv file
    #with open(folder_name + '/temp_chain.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    for rows in temp_chain:
    #        writer.writerow(rows)

    #with open(folder_name + '/value_chain.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    for rows in value_chain:
    #        writer.writerow(rows)

    #with open(folder_name + '/best_value_chain.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    for rows in best_value_chain:
    #        writer.writerow(rows)

    # also record reheat_time and best_time_step
    reheat_time = np.array(reheat_time.reshape(self.config.num_models*self.config.batch_size).transpose())
    reheat_time = list(reheat_time)

    with open(folder_name + '/reheat_time.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rows in reheat_time:
            writer.writerow([rows])
# Improving the aesthetics of the 3D plot and adding gradient arrows
def new_grad_f_3d(x, y):
    df_dx = -2 * x + 0.75
    df_dy = 2 * y - 1
    return df_dx, df_dy

# Create a new 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface with a more refined colormap and transparency
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')

# Annotating points and adding gradient arrows
for point in points:
    value = new_f_3d(*point)
    grad = new_grad_f_3d(*point)
    ax.scatter(*point, value, color='black', s=50)  # Increase the size of the points
    ax.quiver(*point, value, grad[0]/20, grad[1]/20, 0, length=0.1, normalize=True, color='green', arrow_length_ratio=0.5)

# Adjusting colorbar and labels for clarity
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.ax.tick_params(labelsize=8)  # Adjusting colorbar tick size

# Adjusting labels and title for aesthetics
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_zlabel('$f(x, y)$', fontsize=12)
ax.set_title('3D Graph of $f(x, y) = y^2 - y - x^2 + 3/4x$', fontsize=14, pad=20)

# Fine-tuning plot aesthetics
ax.view_init(elev=25, azim=-45)  # Adjusting the viewing angle
ax.dist = 10  # Adjusting the distance of the viewer to the plot

plt.show()

    if reheat_type == 'Best_Temp':
      Best_Temp_Step = np.array(Best_Temp_Step.transpose())
      Best_Temp_Step = list(Best_Temp_Step)
      with open(folder_name + '/Best_Temp_Step.csv', 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)
          for rows in Best_Temp_Step:
              writer.writerow(rows)

    if reheat_type == 'specific_heat' and False:
      #specific_heat_chain = specific_heat_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size)
      #specific_heat_chain = np.array(specific_heat_chain)
      #specific_heat_chain = list(specific_heat_chain)
      #with open(folder_name + '/specific_heat_chain.csv', 'w', newline='') as csvfile:
      #    writer = csv.writer(csvfile)
      #    for rows in specific_heat_chain:
      #        writer.writerow(rows)

      reheat_step = np.array(reheat_step.reshape(self.config.num_models*self.config.batch_size).transpose())
      reheat_step = list(reheat_step)
      with open(folder_name + '/reheat_step.csv', 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)
          for rows in reheat_step:
              writer.writerow([rows])
    '''
    #raise ValueError('Stop here')
    if not (self.config.save_samples or self.config_model.name == 'normcut'):
      best_samples = []
    #saver.save_co_resuts(
    #    chain, best_ratio[sample_mask], running_time, best_samples
    #)
    saver.save_co_resuts(
        [], [], 0, best_samples
    )
    saver.save_results(acc_ratios, hops, None, running_time)


  def _compute_chain_homo(
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

    stp_burnin, stp_mixing, get_hop, obj_fn, _ = compiled_fns
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    specific_heat_chain =  jnp.zeros((self.config.chain_length, self.config.num_models, self.config.batch_size))
    value_chain = jnp.zeros((self.config.chain_length, self.config.num_models ,self.config.batch_size))

    reheat = False
    fake_step = jnp.ones((self.config.num_models, self.config.batch_size), dtype=jnp.int32)
    temp_chain = jnp.zeros((self.config.chain_length, self.config.num_models, self.config.batch_size))
    if reheat:
      trapped_judge = jnp.ones((self.config.num_models, self.config.batch_size), dtype=jnp.int32) * 1e-5
      reheat_step = jnp.ones((self.config.num_models, self.config.batch_size), dtype=jnp.int32)
      max_specific_heat = jnp.zeros((self.config.num_models, self.config.batch_size), dtype=jnp.float32)

    temp_shape = bshape+(self.config.batch_size,)
    init_temperature = jnp.ones(temp_shape, dtype=jnp.float32)
    params['temperature'] = t_schedule(0) * init_temperature
    inner_chain_length = 100

    for step in tqdm.tqdm(range(1, 1 + self.config.chain_length)):
      temp = t_schedule(fake_step)
      temp = jnp.array(temp).reshape(params['temperature'].shape)
      params['temperature'] = temp
      append_temp = temp.reshape(self.config.num_models, self.config.batch_size)
      temp_chain = temp_chain.at[step-1].set(append_temp)
      value = jnp.zeros((inner_chain_length, self.config.num_models, self.config.batch_size))
      for homo_step in range(inner_chain_length):
        rng = jax.random.fold_in(rng, step*self.config.chain_length + homo_step)
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
        value = value.at[homo_step].set(eval_val)
        x = new_x

      inter_value = value.reshape(inner_chain_length, -1).transpose()
      var_value = jnp.var(inter_value, 1)
      inter_temp = append_temp.reshape(var_value.shape)
      specific_heat = var_value / (inter_temp ** 2)  # shape is num_models * batch_size
      specific_heat = specific_heat.reshape((self.config.num_models, self.config.batch_size))
      specific_heat_chain = specific_heat_chain.at[step - 1].set(specific_heat)
      value_chain = value_chain.at[step - 1].set(obj_fn(samples=x, params=params).reshape(self.config.num_models, -1))

      if reheat:
        fake_step = jnp.where(jnp.abs(value_chain[step-1] - value_chain[step-2])>1e-3, fake_step + jnp.ones_like(fake_step), reheat_step)
        if step >= 5:
          max_specific_heat = jnp.maximum(max_specific_heat, specific_heat)
          reheat_step = jnp.where(max_specific_heat == specific_heat, fake_step, reheat_step)
      else:
        fake_step = fake_step + jnp.ones_like(fake_step)


    # save reheat_saver as a csv file, each list writing as a column
    import csv
    import os
    import matplotlib.pyplot as plt

    folder_name = 'Running Results'
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)

    temp_chain = temp_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size)
    temp_chain = np.array(temp_chain)
    temp_chain = list(temp_chain)
    value_chain = value_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size)
    value_chain = np.array(value_chain)
    value_chain = list(value_chain)
    specific_heat_chain = specific_heat_chain.reshape(self.config.chain_length, self.config.num_models*self.config.batch_size)
    specific_heat_chain = np.array(specific_heat_chain)
    specific_heat_chain = list(specific_heat_chain)

    # save temp_chain and value_chain as csv file
    with open(folder_name + '/temp_chain.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rows in temp_chain:
            writer.writerow(rows)

    with open(folder_name + '/value_chain.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rows in value_chain:
            writer.writerow(rows)

    with open(folder_name + '/specific_heat_chain.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rows in specific_heat_chain:
            writer.writerow(rows)

    #reheat_step = np.array(reheat_step.reshape(self.config.num_models*self.config.batch_size).transpose())
    #reheat_step = list(reheat_step)
    #with open(folder_name + '/reheat_step.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    for rows in reheat_step:
    #        writer.writerow([rows])

    raise ValueError('Done')









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


class EBM_Experiment(Experiment):

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    """Initializes model params, sampler state and gets the initial samples."""
    rng_param, rng_x, rng_state = jax.random.split(rnd, num=3)
    del rnd
    params = model.make_init_params(rng_param)
    params['temperature'] = 0
    x = model.get_init_samples(rng_x, self.config.batch_size)
    state = sampler.make_init_state(rng_state)
    return params, x, state

  def _compile_fns(self, sampler, model):
    if not self.parallel:
      score_fn = jax.jit(model.forward)
      step_fn = jax.jit(functools.partial(sampler.step, model=model))
    else:
      score_fn = jax.pmap(model.forward)
      step_fn = jax.pmap(functools.partial(sampler.step, model=model))
    return (
        score_fn,
        step_fn,
    )

  def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
    rnd = jax.random.PRNGKey(rnd_key)
    params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
    if self.parallel:
      params = jax.device_put_replicated(params, jax.local_devices())
      state = jax.device_put_replicated(state, jax.local_devices())
      assert self.config.batch_size % jax.local_device_count() == 0
      nn = self.config.batch_size // jax.local_device_count()
      x = x.reshape((jax.local_device_count(), nn) + self.config_model.shape)
    compiled_fns = self._compile_fns(sampler, model)
    return [
        compiled_fns,
        state,
        params,
        rnd,
        x,
        saver,
        model,
    ]

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      model,
  ):
    """Generates the chain of samples."""

    score_fn, stp_fn = compiled_fns

    logz_finals = []
    log_w = jnp.zeros(self.config.batch_size)
    if self.parallel:
      log_w = log_w.reshape(x.shape[0], -1)

    for step in tqdm.tqdm(range(1, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)

      old_val = score_fn(params, x)
      if not self.parallel:
        params['temperature'] = step * 1.0 / self.config.chain_length
      else:
        params['temperature'] = jnp.repeat(
            step * 1.0 / self.config.chain_length, x.shape[0]
        )

      log_w = log_w + score_fn(params, x) - old_val
      if not self.parallel:
        rng_step = rng
      else:
        rng_step = jax.random.split(rng, x.shape[0])
      new_x, state, _ = stp_fn(
          rng=rng_step,
          x=x,
          model_param=params,
          state=state,
      )
      log_w_re = log_w.reshape(-1)
      logz_final = jax.scipy.special.logsumexp(log_w_re, axis=0) - np.log(
          self.config.batch_size
      )
      logz_finals.append(logz_final)
      x = new_x

    saver.save_logz(logz_finals)
