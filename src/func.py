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
from constants import *

x1 = ["aa_000", "bt_000", "bv_000"]
y1 = ["ci_000", "db_000", "ca_000", "cd_000"]

def displayInfo(train):
    print("Header :", train.head(5))
    print("Shape :", train.shape)
    print("Describe :", train.describe())
    print("Info :", train.info())

def displayCorr(train):
    correlations = train.corr()["class"].sort_values(ascending=False)
    print(correlations.to_string())

def pre_processing(df,value_for_na = 0):
    df["class"] = df["class"].map({"neg":0,"pos":1})
    df = df.replace("na",value_for_na)
    for col in df.columns:
        if col != "origin":
            df[col] = pd.to_numeric(df[col])
    return df

def displayHisto(neg_means, pos_means):

    print("Moyennes pour la classe 'neg':")
    print(neg_means)

    print("\nMoyennes pour la classe 'pos':")
    print(pos_means)
    columns = neg_means.index

    fig, ax = plt.subplots()
    bar_width = 0.5
    bar_positions_neg = np.arange(len(columns))
    bar_positions_pos = bar_positions_neg + bar_width

    ax.bar(bar_positions_neg, neg_means, bar_width, label='neg')
    ax.bar(bar_positions_pos, pos_means, bar_width, label='pos')

    ax.set_xticks(bar_positions_neg + bar_width / 2)
    ax.set_xticklabels(columns, rotation=60, ha='right')
    ax.legend()
    ax.set_xlabel('Colonnes')
    ax.set_ylabel('Valeurs moyennes')
    ax.set_title('Moyennes pour chaque colonne par classe')

    plt.tight_layout()
    plt.show()


def knn_classification(train_df, test_df, feature1, feature2, k=5):
    X_train = train_df[[feature1, feature2]]
    y_train = train_df["class"]

    X_test = test_df[[feature1, feature2]]
    y_test = test_df["class"]

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, confusion_mat

def find_best_k(train_df, test_df, feature1, feature2, start_k=1, end_k=20):
    best_k = -1
    best_accuracy = 0

    for k in range(start_k, end_k + 1):
        accuracy, _ = knn_classification(train_df, test_df, feature1, feature2, k)

        print(f"Accuracy for k={k}: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"\nBest k: {best_k} with average accuracy: {best_accuracy}")

def display_knn_results(train_df, test_df, feature1, feature2, k=2):
    accuracy, confusion_mat = knn_classification(train_df, test_df, feature1, feature2, k)

    print(f"Accuracy: {accuracy}")
    print("\nConfusion Matrix:")
    print(confusion_mat)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['No failure', 'Failure'])
    disp.plot(cmap='winter', values_format='d', ax=axs[0])
    axs[0].set_title('Confusion Matrix')

    # Normalized
    normalized_confusion_mat = confusion_mat / confusion_mat.sum(axis=1)[:, np.newaxis]
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=normalized_confusion_mat, display_labels=['No failure', 'Failure'])
    disp_normalized.plot(cmap='summer', values_format='.4f', ax=axs[1])
    axs[1].set_title('Normalized Confusion Matrix')

    plt.tight_layout()
    plt.show()

def df_pca(df, pca):
	X_pca = pca.fit_transform(df[x1]).flatten()
	Y_pca = pca.fit_transform(df[y1]).flatten()
	return pd.DataFrame({'class': df["class"], 'X': X_pca, 'Y': Y_pca})


def plot_scatter(df):
	neg = df[df["class"] == 0]
	pos = df[df["class"] == 1]
	plt.scatter(pos["X"], pos["Y"], color = "r")
	plt.scatter(neg["X"], neg["Y"], color = "b")
	plt.show()

