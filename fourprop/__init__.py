# Copyright (C) 2026 Jyotiprakash Mishra <mail@jyotiprakash.org>
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of fourprop; see LICENSE for the terms and NOTICE for the attribution requirement.
"""fourprop: order-k p-biased Fourier propagation of signal probabilities through logic circuits."""
from .circuit import Circuit, parse_bench
from .fourier import propagate, State, product, tail_mass
from .exact import exact_probabilities, exact_coefficients
__all__ = ["Circuit", "parse_bench", "propagate", "State", "product", "tail_mass",
           "exact_probabilities", "exact_coefficients"]
