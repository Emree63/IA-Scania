import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


def display_info(train):
    print("Header :", train.head(5))
    print("Shape :", train.shape)
    print("Describe :", train.describe())
    print("Info :", train.info())

def display_corr(train):
    correlations = train.corr()["class"].sort_values(ascending=False)
    print(correlations.to_string())

def pre_processing(df,value_for_na = 0):
    df["class"] = df["class"].map({"neg":0,"pos":1})
    df = df.replace("na",value_for_na)
    for col in df.columns:
        if col != "origin":
            df[col] = pd.to_numeric(df[col])
    return df

def display_hist(neg_means, pos_means):

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

def display_random_forest_results(accuracy, confusion_mat):
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

def random_forest_classification(train_df, test_df, n_estimators = 7):
    X_train = train_df.drop("class", axis=1).drop("origin", axis=1)
    y_train = train_df["class"]

    X_test = test_df.drop("class", axis=1).drop("origin", axis=1)
    y_test = test_df["class"]

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion="entropy", 
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=20
        )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    display_random_forest_results(accuracy, confusion_mat)

def find_best_model(train_df, test_df):
    X_train = train_df.drop("class", axis=1).drop("origin", axis=1)
    y_train = train_df["class"]

    X_test = test_df.drop("class", axis=1).drop("origin", axis=1)
    y_test = test_df["class"]

    param_grid = { 
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RandomForestClassifier(n_estimators=15, random_state=42)
    CV_rfc = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)

    # Best hyperparameters
    best_params = CV_rfc.best_params_
    print("Best Hyperparameters:", best_params)

    # Best model
    best_rf_model = CV_rfc.best_estimator_

    best_rf_model.fit(X_train, y_train)

    y_pred = best_rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    display_random_forest_results(accuracy, confusion_mat)

def find_best_n(train_df, test_df, start_n=1, end_n=20):
    best_n = -1
    best_accuracy = 0

    for n in range(start_n, end_n + 1):
        accuracy, _ = random_forest_classification(train_df, test_df, n)

        print(f"Accuracy for n={n}: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n = n

    print(f"\nBest n: {best_n} with average accuracy: {best_accuracy}")

