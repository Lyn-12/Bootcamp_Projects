import pandas as pd
import pyforest
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, log_loss, classification_report)
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
##python3 -m venv sane-env  - create a virtual environment
#source sane-env/bin/activate - activate the virtual environment
##python3 -m Week2_Project.Airbnb_prediction

## load the dataset
df = pd.read_csv('/Users/lynxy/Desktop/Lux_Datascience/Data Science_Bootcamp_Projects/Week2_Project/train.csv', sep = ',')
#print(df.columns)

## Explore the dataset
print(df.info())
print(df.describe())
print(df.shape)
## The data has 74111 rows x 29 columns
print(df.head())