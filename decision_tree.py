import random

# Parse data from file:
data_file = open("car.data", 'r')
db = [line.split(',') for line in data_file.read().splitlines()]

# Split database into training data and test data:
db_size = len(db)
db_training_size = int(db_size * 0.2)

random.shuffle(db)
db_training = db[db_training_size:]
db_test = db[:db_training_size]

# Naive Bayesian Classification:



