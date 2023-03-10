"""
paths.py
====================================
Module to hold paths of files.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')