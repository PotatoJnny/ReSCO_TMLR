#!/bin/bash
export XLA_FLAGS='--xla_force_host_platform_device_count=4'

default="default_value"
graph_type=${graph_type:-$default}
echo "$model"
echo "$sampler"
echo "$graph_type"


if [ "$graph_type" == "$default" ]
then
   exp_config="ReSCO/common/configs.py"
else
   exp_config="ReSCO/experiment/configs/${model?}/${graph_type:-$default}.py" 
fi

if [ "$model" == "text_infilling" ]
then
   exp_config="ReSCO/experiment/configs/lm_experiment.py"
fi

echo $exp_config


python -m ReSCO.experiment.main_sampling \
  --model_config="ReSCO/models/configs/${model?}_config.py" \
  --sampler_config="ReSCO/samplers/configs/${sampler?}_config.py" \
  --config=$exp_config \
  --run_local=True \


