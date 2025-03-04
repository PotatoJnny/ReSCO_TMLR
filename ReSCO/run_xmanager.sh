#!/bin/bash

xmanager launch ReSCO/xm_launcher.py -- \
--config=${config?} \
--xm_resource_alloc="user:xcloud/xcloud-shared-user" \

