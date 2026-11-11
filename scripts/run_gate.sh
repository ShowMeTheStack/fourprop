#!/usr/bin/env bash
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
# Cron/@reboot entry point: idempotent, single-instance, resumable.
ROOT=/extra/fourprop
GATE=${JOBBASE:-$ROOT}/gate
mkdir -p $GATE/logs
[ -f $GATE/DONE ] && exit 0
exec flock -n $GATE/runner.lock \
  nice -n 5 $ROOT/venv/bin/python $ROOT/repo/exp/gate_runner.py \
    --root $GATE --bench $ROOT/bench/iscas85 --workers 12 >> $GATE/logs/cron.log 2>&1
