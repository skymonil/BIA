# 1. Import libraries
from sklearn.linear_model import LogisticRegression
import numpy as pd, pandas as pd
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

