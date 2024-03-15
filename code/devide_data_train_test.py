import random

with open('C:/Users/plug2/Desktop/diploma/citeulike-a-master/users.dat', 'r') as file:
    lines = file.readlines()

# Define the proportions for train and test
train_proportion = 0.8
test_proportion = 0.2

# Initialize lists to store the train and test data
train_data = []
test_data = []

# Iterate over each line and split into train and test
for line in lines:
    items = line.strip().split()[1:]  # Split the line into items
    random.shuffle(items)  # Shuffle the items randomly
    
    # Determine the number of items for train and test based on proportions
    num_train = int(len(items) * train_proportion)
    num_test = len(items) - num_train
    
    # Append the train and test items to respective lists
    train_data.append(' '.join(items[:num_train]) + '\n')
    test_data.append(' '.join(items[num_train:]) + '\n')

# Save the train and test data into new .dat files
with open('C:/Users/plug2/Desktop/diploma/citeulike-a-master/users_80.dat', 'w') as train_file:
    train_file.writelines(train_data)

with open('C:/Users/plug2/Desktop/diploma/citeulike-a-master/users_20.dat', 'w') as test_file:
    test_file.writelines(test_data)
