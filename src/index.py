from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib.patches as mpatches
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from func import *
from constants import *

# Init
train = pd.read_csv(TRAINING_PATH)
test = pd.read_csv(TEST_PATH)

train["origin"] = "train"
test["origin"] = "test"

train = pre_processing(train)
test = pre_processing(test)

neg_data = train[train["class"] == 0]
pos_data = train[train["class"] == 1]

neg_means = neg_data.drop(columns=["class", "origin"]).mean()
pos_means = pos_data.drop(columns=["class", "origin"]).mean()

pca = PCA(n_components=0.95)

#train_pca, test_pca = df_pca(train), df_pca(test)

def menu():
    print("1. Display Histogramme")
    print("2. Display Infos")
    print("3. Display KNN results")
    print("4. Find best K value")
    print("5. Perform PCA and plot scatter")
    print("6. Perform PCA and KNN classification")
    print("9. Exit")

    choice = input("Enter your choice: ")

    return int(choice)

def execute_function(choice, train, test, x_feature, y_feature):
    if choice == 1:
        displayHisto(neg_means, pos_means)
    elif choice == 2:
        print("Train :")
        displayInfo(train)
        print("Test :")
        displayInfo(test)
    elif choice == 3:
        display_knn_results(train, test, x_feature, y_feature)
    elif choice == 4:
        find_best_k(train, test, x_feature, y_feature)
    #elif choice == 5:
        #plot_scatter(train_pca)
    #elif choice == 6:
        #knn_classification(train_pca, test_pca, x_feature, y_feature)
        #find_best_k(train_pca, test_pca, x_feature, y_feature)

choice = menu()

while choice != 9:
    execute_function(choice, train, test, "ac_000", "cq_000")
    choice = menu()


# x1 = ["ab_000", "bb_000", "bv_000", "bu_000"]
# y1 = ["dq_000", "cq_000", "cc_000"]

x1 = ["ab_000"]
y1 = ["dq_000", "bv_000", "cq_000"]

#plot_scatter(train_pca)
#knn_classification(train_pca, test_pca, "X", "Y")
#find_best_k(train_pca, test_pca, "X", "Y")