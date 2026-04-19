"""Backends sub-package for SSD fitting algorithms.

Contains three fitting backends (``pls``, ``pca_sweep``, ``group``) plus
shared math helpers (``_sweep_math``).  Each backend is imported on demand
by :class:`~ssdiff.ssd.SSD` to keep startup time low.
"""
