import os
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
ID = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# numerical and scientific libs
import pandas as pd
import numpy as np
import math
import scipy 
from scipy import ndimage


# image libs
import cv2 
import PIL
from PIL import Image

# fastai imports
from fastai import * 
from fastai.conv_learner import *
from fastai.dataset import *

# viz
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.graph_objs as go
matplotlib.use('TkAgg')

# other libs
import collections
import csv
from shutil import copy
from sklearn.model_selection import train_test_split
from skimage.transform import resize as skimage_resize
from tqdm import tqdm, tnrange
import time

import warnings
warnings.filterwarnings('ignore')

# constants 
PI = np.pi
INF = np.inf
EPS = 1e-7
random_state = 42

# dirs
DATA_DIR = './data'
SEGMENT = './data/train_ship_segmetations_v2.csv'
TRAIN = './data/train_v2'
TEST = './data/test_v2'



