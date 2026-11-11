#!/usr/bin/env bash
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
# Clean-box timing job: only starts when the 1-minute load average is below MAXLOAD; cron retries every 10 min.
ROOT=/extra/fourprop; JOB=${JOBBASE:-$ROOT}/timing; MAXLOAD=${MAXLOAD:-6}; mkdir -p $JOB/logs
[ -f $JOB/DONE ] && exit 0
LOAD=$(cut -d" " -f1 /proc/loadavg)
if [ "$(echo "$LOAD > $MAXLOAD" | bc -l)" = "1" ]; then echo "$(date) load $LOAD > $MAXLOAD, waiting" >> $JOB/logs/wait.log; exit 0; fi
exec flock -n $JOB/runner.lock $ROOT/venv/bin/python $ROOT/repo/exp/timing_runner.py --root $JOB --bench $ROOT/bench >> $JOB/logs/cron.log 2>&1
