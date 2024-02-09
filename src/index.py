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

def menu():
    print("1. Display Histogram")
    print("2. Display Correlation")
    print("3. Display Infos")
    print("4. Display Random Forest results")
    print("5. Find best model")
    print("\n9. Exit")

    choice = input("Enter your choice: ")

    return int(choice)

def execute_function(choice, train, test, x_feature, y_feature):
	if choice == 1:
		display_hist(neg_means, pos_means)
	elif choice == 2:
		display_corr(train)
	elif choice == 3:
		print("Train :")
		display_info(train)
		print("Test :")
		display_info(test)
	elif choice == 4:
		random_forest_classification(train, test, int(input("N: ")))
	elif choice == 5:
		find_best_model(train, test)

choice = menu()

while choice != 9:
    execute_function(choice, train, test, "ac_000", "cq_000")
    choice = menu()