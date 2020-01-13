import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
import imageio
import collections
import pickle
import re
import requests
import pandas as pd
from pathlib import Path
import pretty_errors
from filter_BU import filt_B
import my_pyrotd
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
from post_processing import *
InteractiveShell.ast_node_interactivity = "last_expr"
# InteractiveShell.ast_node_interactivity = "all"

get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic("matplotlib", " inline")
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
rcparams = {'font.size': 16,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'lines.linewidth': 1.5,
            'figure.dpi': 300}
plt.rcParams.update(rcparams)
plt.style.use('seaborn')
