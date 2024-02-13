import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

def pre_processing(df, value_for_na = 0):
    """!
    Processes the dataframe to make it easier to handle.

    @param df (pd.DataFrame): The input DataFrame.
    @param value_for_na (int, optional): Value to replace 'na' with (default is 0).

    @return pd.DataFrame: Processed DataFrame.
    """    

    # Replaces 'pos' and 'neg' values with 1 and 0
    df["class"] = df["class"].map({"neg":0,"pos":1})

    # Replaces 'na' values with value_for_na (0 by default)
    df = df.replace("na", value_for_na)

    # Converts the dataframe values into numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def display_info(df):
    """!
    Displays stats about the dataframe.

    @param df (pd.DataFrame): The input DataFrame.
    """

    print("Header :", df.head(5))
    print("Shape :", df.shape)
    print("Describe :", df.describe())
    print("Info :", df.info())

def display_corr(df):
    """!
    Displays correlation of columns with the 'class' column and a bar plot.

    @param df (pd.DataFrame): The input DataFrame.
    """

    correlations = df.corr()["class"].sort_values(ascending=False)
    print(correlations.to_string())

    correlations[1:12] .plot(kind='bar')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.title('Correlation with Target Class')
    plt.show()

def random_forest_classification(train_df, test_df, n_estimators = 7):
    """!
    Apply RandomForestClassifier to test_df and return accuracy and confusion matrix.

    @param train_df (pd.DataFrame): Training DataFrame.
    @param test_df (pd.DataFrame): Testing DataFrame.
    @param n_estimators (int, optional): Number of trees in the forest (default is 7).
    """

    # Isolate the data and the labels
    X_train, y_train = train_df.drop("class", axis=1), train_df["class"]
    X_test, y_test = test_df.drop("class", axis=1), test_df["class"]

    # Creates the model, here we use a RandomForestClassifier with parameters chosen using the find_best_model function
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion="entropy", 
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=20
    )


    # Fit the model on the training data
    rf_model.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = rf_model.predict(X_test)

    # Get and display results
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    display_random_forest_results(accuracy, confusion_mat)


def display_random_forest_results(accuracy, confusion_mat):
    """!
    Display accuracy and confusion matrix in a formatted way and plot confusion matrices.

    @param accuracy (float): Classification accuracy.
    @param confusion_mat (np.ndarray): Confusion matrix.
    """

    print(f"Accuracy: {accuracy}")
    print("\nConfusion Matrix:")
    print(confusion_mat)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['No failure', 'Failure'])
    disp.plot(cmap='winter', values_format='d', ax=axs[0])
    axs[0].set_title('Confusion Matrix')

    # Normalize the matrix
    normalized_confusion_mat = confusion_mat / confusion_mat.sum(axis=1)[:, np.newaxis]
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=normalized_confusion_mat, display_labels=['No failure', 'Failure'])
    disp_normalized.plot(cmap='summer', values_format='.4f', ax=axs[1])
    axs[1].set_title('Normalized Confusion Matrix')

    plt.tight_layout()
    plt.show()


def find_best_model(train_df, test_df):
    """!
    Find the best RandomForestClassifier Model with a parameter grid.

    @param train_df (pd.DataFrame): Training DataFrame.
    @param test_df (pd.DataFrame): Testing DataFrame.
    """

    # Isolate the data and the labels
    X_train, y_train = train_df.drop("class", axis=1), train_df["class"]
    X_test, y_test = test_df.drop("class", axis=1), test_df["class"]

    # Define the parameter grid
    param_grid = { 
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        # 'max_depth': [None, 10, 20],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4]
    }


    # Creates a RandomForestClassifier model
    rf_model = RandomForestClassifier(n_estimators=15, random_state=42)

    # Inits a GridSearchCV with the model and the parameter grid to find the best working parameters
    CV_rfc = GridSearchCV(estimator=rf_model, param_grid=param_grid)

    # Fit the model on the training data
    CV_rfc.fit(X_train, y_train)

    # Get best hyperparameters and prints them
    best_params = CV_rfc.best_params_
    print("Best Hyperparameters:", best_params)

    # Get the best working model
    best_rf_model = CV_rfc.best_estimator_

    # Fit the best model on the training data
    best_rf_model.fit(X_train, y_train)

    # Predict the labels for the test data with the best model
    y_pred = best_rf_model.predict(X_test)

    # Get and display results
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    display_random_forest_results(accuracy, confusion_mat)
