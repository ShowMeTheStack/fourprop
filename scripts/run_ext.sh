#!/usr/bin/env bash
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
# Extension pipeline: ablation -> pareto -> cv -> errmodel under one JOBBASE (default /extra/fourprop/ext).
# Resumable; install as cron: @reboot sleep 300 && JOBBASE=... run_ext.sh ; */10 * * * * same.
ROOT=/extra/fourprop
export JOBBASE=${JOBBASE:-$ROOT/ext}
mkdir -p $JOBBASE/logs
exec 9>$JOBBASE/run_ext.lock; flock -n 9 || exit 0
echo "$(date) run_ext start (JOBBASE=$JOBBASE, load $(cut -d' ' -f1 /proc/loadavg))" >> $JOBBASE/logs/run_ext.log
for job in ablation pareto cv errmodel; do
  [ -f $JOBBASE/$job/DONE ] && continue
  echo "$(date) starting $job" >> $JOBBASE/logs/run_ext.log
  $ROOT/repo/scripts/run_$job.sh
  if [ ! -f $JOBBASE/$job/DONE ]; then
    echo "$(date) $job did not reach DONE (see $JOBBASE/$job/logs)" >> $JOBBASE/logs/run_ext.log; exit 1
  fi
done
echo "$(date) run_ext complete" >> $JOBBASE/logs/run_ext.log
touch $JOBBASE/EXT_DONE
