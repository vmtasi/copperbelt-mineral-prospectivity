"""DEPRECATED - do not put this directory on PYTHONPATH.

Installing the pyarrow block from a ``sitecustomize`` module corrupts the
QGIS Pillow extension import in this interpreter
(``ImportError: DLL load failed while importing _imaging``).

The supported bootstrap is ``src/runtime_shim.py``, imported explicitly by
each pipeline script immediately after the repository root is placed on
``sys.path``.  This file is retained only so that its history is not lost.
"""
