#!/usr/bin/env bash
# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
# Fetch every summary (markdown + rows json) of a job base to a local directory.
#   scripts/collect.sh /extra/fourprop/run2 ~/rhiddhi/algebra/results_run2
REMOTE=${1:-/extra/fourprop}; LOCAL=${2:-./results}; mkdir -p "$LOCAL"
for j in gate fast proj phase2 phase2b oracle power justify justify_hard justify_hardr timing; do
  rsync -az --include='*.md' --include='*rows.json' --exclude='*' "mdev@mdev-desktop:$REMOTE/$j/" "$LOCAL/$j/" 2>/dev/null && echo "fetched $j"
done
