###
# File Name: naive_bayesian.py
# Created By: Team Mockingjay
# Created Date: 11/28/2015
# Last Edit: 11/28/2015
#

# Import libraries
import math

# Initialize the attributes list
# (source: car.c45-names)
attributes = [['vhigh', 'high', 'med', 'low'], 
				['vhigh', 'high', 'med', 'low'], 
				['2', '3', '4', '5more'], 
				['2', '3', '4', 'more'],
				['small', 'med', 'big'], 
				['low', 'med', 'high']]

# Function: transform the given value to one hot encoding based on the configuration of feature class
# eg: if the value is 'vhigh', given the feature class being ['vhigh', 'high', 'med', 'low']
#      the one hot code for it will be 1000
def oneHotCode(value, featureClass):
	oneHotCode = ['0']*len(featureClass)
	oneHotCode[featureClass.index(value)] = '1'
	return ','.join(oneHotCode)
	
# Function: transform a database entry in to one hot encoding
def oneHotEntry(oldEntry, attributes):
	oneHotEntry=''
	for index, value in enumerate(oldEntry):
		oneHotEntry += oneHotCode(value,attributes[index])+','
	return oneHotEntry

# Function: transform the whole data base into one hot encoding
def oneHotTransformation(db, attributes):
	oneHotDb = []
	for entry in db:
		oneHotDb.append(oneHotEntry(entry[:-1], attributes)+entry[-1])
	return oneHotDb

if __name__ == "__main__":
	# Parse training and testing data from file:
	train_file = open("train.data", 'r')
	test_file = open("test.data", 'r')
	db_train = [line.split(',') for line in train_file.read().splitlines()]
	db_test = [line.split(',') for line in test_file.read().splitlines()]
	train_file.close()
	test_file.close()

	oneHotDb_train = oneHotTransformation(db_train, attributes)
	oneHotDb_test = oneHotTransformation(db_test, attributes)

	oneHotDb_train_file = open("oneHot_train.data",'w')
	oneHotDb_test_file = open("oneHot_test.data", 'w')

	for oneHotEntry in oneHotDb_train:
		oneHotDb_train_file.write(oneHotEntry+'\n')

	for oneHotEntry in oneHotDb_test:
		oneHotDb_test_file.write(oneHotEntry+'\n')




