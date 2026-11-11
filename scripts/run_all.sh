#!/usr/bin/env bash
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
# Run every experiment job in sequence under one job base (default: a fresh "runN" directory).
#   JOBBASE=/extra/fourprop/run2 scripts/run_all.sh
# Each job is resumable (result files + DONE marker); rerunning this script after a reboot continues where it stopped.
# Install as cron: @reboot sleep 300 && JOBBASE=... /extra/fourprop/repo/scripts/run_all.sh ; */10 * * * * same.
ROOT=/extra/fourprop
export JOBBASE=${JOBBASE:-$ROOT/run2}
mkdir -p $JOBBASE/logs
exec 9>$JOBBASE/run_all.lock; flock -n 9 || exit 0
echo "$(date) run_all start (JOBBASE=$JOBBASE, load $(cut -d' ' -f1 /proc/loadavg))" >> $JOBBASE/logs/run_all.log
for job in gate fast proj phase2 phase2b oracle power justify justify_hard justify_hardr timing; do
  [ -f $JOBBASE/$job/DONE ] && continue
  echo "$(date) starting $job" >> $JOBBASE/logs/run_all.log
  $ROOT/repo/scripts/run_$job.sh          # blocks until the job's runner exits (DONE or failure)
  if [ ! -f $JOBBASE/$job/DONE ]; then
    echo "$(date) $job did not reach DONE (timing waits for low load; others: see $JOBBASE/$job/logs)" >> $JOBBASE/logs/run_all.log
    [ "$job" = timing ] && exit 0 || exit 1
  fi
done
echo "$(date) run_all complete" >> $JOBBASE/logs/run_all.log
touch $JOBBASE/ALL_DONE
