__author__ = "liuwei"

"""
the train file of word-based Chinese NER model
"""

import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import re
import os

