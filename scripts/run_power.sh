#!/usr/bin/env bash
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
ROOT=/extra/fourprop; JOB=${JOBBASE:-$ROOT}/power; mkdir -p $JOB/logs
[ -f $JOB/DONE ] && exit 0
exec flock -n $JOB/runner.lock nice -n 5 $ROOT/venv/bin/python $ROOT/repo/exp/power_runner.py \
  --root $JOB --bench $ROOT/bench --workers 6 >> $JOB/logs/cron.log 2>&1
