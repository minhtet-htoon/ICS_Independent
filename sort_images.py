import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.chdir('data/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for i in random.sample(glob.glob('cat*'), 1000):
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 1000):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 200):
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 200):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'test/dog')

os.chdir('../../')