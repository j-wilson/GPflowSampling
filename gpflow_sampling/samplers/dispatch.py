#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gpflow.utilities import Dispatcher

__all__ = ('location_scale', 'finite_fourier', 'decoupled')

location_scale = Dispatcher("location_scale")
finite_fourier = Dispatcher("finite_fourier")
decoupled = Dispatcher("decoupled")

