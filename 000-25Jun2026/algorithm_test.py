from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import plotly.express as px
from PIL import ImageTk, Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
