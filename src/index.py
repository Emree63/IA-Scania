from func import *
from constants import *

# Init
train = pd.read_csv(TRAINING_PATH)
test = pd.read_csv(TEST_PATH)

train = pre_processing(train)
test = pre_processing(test)

def menu():
    print("1. Display Correlation")
    print("2. Display Infos")
    print("3. Display Random Forest results")
    print("4. Find best model")
    print("\n9. Exit")

    choice = input("Enter your choice: ")

    return int(choice)

def execute_function(choice, train, test, x_feature, y_feature):
	if choice == 1:
		display_corr(train)
	elif choice == 2:
		print("Train :")
		display_info(train)
		print("Test :")
		display_info(test)
	elif choice == 3:
		random_forest_classification(train, test, int(input("N: ")))
	elif choice == 4:
		find_best_model(train, test)

choice = menu()

while choice != 9:
    execute_function(choice, train, test, "ac_000", "cq_000")
    choice = menu()