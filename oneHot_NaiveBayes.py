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

classes = ['unacc', 'acc', 'good', 'vgood']
# Function: Compute Info(p, n)
def info(p, n):
	a = float(p)/(p+n)
	b = float(n)/(p+n)
	return -a * math.log(a,2) - b * math.log(b,2)

# Function: Return a list of objects having a specific value
def filter_db(db, val, attr_index):
	return [t for t in db if t[attr_index] == val]

# Function: Returns the probability that an attribute at index=attr_index
# 			in a dataset db is equal to val
def attr_prob(db, val, attr_index):
	return sum(bool(t[attr_index] == val) for t in db) / float(len(db))

class nb_classifier:
	# Initilize the classifier
	def __init__(self, db):
		self.db = db
		self.attributes_length = len(db[0])-1
		self.cond_prob = self.compute_cond_prob()
		self.class_prob = self.compute_class_prob()

	# Compute all conditional probabilities: 
	# Output: 
	#	prob['acc'][0]['high'] = P(buying = 'high' | 'acc') 
	#	i.e. prob['acc'][0]['high'] denotes: 
	#		probability that the attribute at index 0 ('buying')
	#		is 'high' given an 'acc' class
	def compute_cond_prob(self):
		prob = {}
		for cls in classes:
			db_proj = filter_db(self.db, cls, -1)
			prob[cls] = []
			for i in range(0, self.attributes_length):
				prob[cls].append({'1':attr_prob(db_proj, '1', i), '0':attr_prob(db_proj, '0', i)})
				# Use a small default probability (0.001)
				# 	in case some <feature value, class> combination
				#	is absent in the training dataset
				if prob[cls][i]['1'] == 0:
					prob[cls][i]['1'] = 0.001
				if prob[cls][i]['0'] == 0:
					prob[cls][i]['0'] = 0.001
		#print prob
		return prob

	# Compute all class probabilities: 
	# Output: 
	#	prob['acc'] = P('acc') 
	def compute_class_prob(self):
		prob = {}
		for cls in classes:
			prob[cls] = attr_prob(self.db, cls, -1)
		#print prob
		return prob

	# Predict the tuple's class
	def classify(self, tuple):
		prob = {}
		# Iterate through 4 classes
		for cls in classes:
			# Log transform
			prob_tmp = math.log(self.class_prob[cls])
			# Iterate through each attribute
			for i in range(0,self.attributes_length):
				attr_val = tuple[i]
				
				# Use log to transform multiplications into additions
				try:
					prob_tmp = prob_tmp + math.log(self.cond_prob[cls][i][attr_val])
				except:
					print self.cond_prob[cls][i][attr_val]
			# Here prob[clus] is the log-transformed probability
			prob[cls] = prob_tmp
		#print prob
		return max(prob, key=prob.get)

		
# Main function
if __name__ == "__main__":
	# Parse training and testing data from file:
	train_file = open("oneHot_train.data", 'r')
	test_file = open("oneHot_test.data", 'r')
	db_train = [line.split(',') for line in train_file.read().splitlines()]
	db_test = [line.split(',') for line in test_file.read().splitlines()]
	train_file.close()
	test_file.close()

	# Naive Bayesian Classification:
	nbc = nb_classifier(db_train)

	# Compute the accuracy of the classifier
	accuracy_train = sum(bool(t[-1] == nbc.classify(t)) for t in db_train) / float(len(db_train))
	accuracy_test = sum(bool(t[-1] == nbc.classify(t)) for t in db_test) / float(len(db_test))
	print "Accuracy of NB Classifier:"
	print "	- Training dataset:", accuracy_train
	print "	- Test dataset:", accuracy_test, '\n'

	# Confusion Matrix:
	conf_mat = {'unacc':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}, 
				'acc':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}, 
				'good':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}, 
				'vgood':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}}

	# Construct confusion matrix
	for t in db_test:
		actual_class = t[-1]
		predicted_class = nbc.classify(t)
		conf_mat[actual_class][predicted_class] += 1

	# Output confusion matrix
	print "Confusion Matrix: "
	print conf_mat



