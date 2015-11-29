from __future__ import print_function
import random

# Parse data from file:
data_file = open("car.data", 'r')
db = [line.split(',') for line in data_file.read().splitlines()]
data_file.close()

# Split database into training data and test data:
db_size = len(db)
db_training_size = int(db_size * 0.2)

random.shuffle(db)
db_train = db[db_training_size:]
db_test = db[:db_training_size]

train_file = open("train.data", 'w')
for t in db_train:
	train_file.write(t[0]+","+t[1]+","+t[2]+","+t[3]+","+t[4]+","+t[5]+","+t[6]+'\n')
train_file.close()

test_file = open("test.data", 'w')
for t in db_test:
	test_file.write(t[0]+","+t[1]+","+t[2]+","+t[3]+","+t[4]+","+t[5]+","+t[6]+'\n')
test_file.close()