from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib.patches as mpatches
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

from constants import *

train = pd.read_csv(TRAINING_PATH)
test = pd.read_csv(TEST_PATH)

def displayInfo():
    print("Header :", train.head(5))
    print("Shape :", train.shape)
    print("Describe :", train.describe())
    print("Info :", train.info())

    print("Header :", test.head(5))
    print("Shape :", test.shape)
    print("Describe :", test.describe())
    print("Info :", test.info())


def pre_processing(df,value_for_na = -1):
    df["class"] = df["class"].map({"neg":0,"pos":1})
    df = df.replace("na",value_for_na)
    for col in df.columns:
        if col != "origin":
            df[col] = pd.to_numeric(df[col])
    return df

train["origin"] = "train"
test["origin"] = "test"

train = pre_processing(train)
test = pre_processing(test)

neg_data = train[train["class"] == 0]
pos_data = train[train["class"] == 1]

neg_means = neg_data.drop(columns=["class", "origin"]).mean()
pos_means = pos_data.drop(columns=["class", "origin"]).mean()

print("Moyennes pour la classe 'neg':")
print(neg_means)

print("\nMoyennes pour la classe 'pos':")
print(pos_means)

def displayHisto():

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