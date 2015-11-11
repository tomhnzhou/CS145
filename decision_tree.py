import random, math

attributes = [['vhigh', 'high', 'med', 'low'], 
				['vhigh', 'high', 'med', 'low'], 
				['2', '3', '4', '5more'], 
				['2', '3', '4', 'more'],
				['small', 'med', 'big'], 
				['low', 'med', 'high'], 
				['unacc', 'acc', 'good', 'vgood']]

# Compute Info(p, n)
def info(p, n):
	a = float(p)/(p+n)
	b = float(n)/(p+n)
	return -a * math.log(a,2) - b * math.log(b,2)

def filter_db(db, val, attr_index):
	return [t for t in db if t[attr_index] == val]

# Returns probability that an attribute at index=attr_index
# 	in a dataset db is equal to val
def attr_prob(db, val, attr_index):
	return sum(bool(t[attr_index] == val) for t in db) / float(len(db))

class nb_classifier:
	def __init__(self, db):
		self.db = db
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
		for cls in attributes[6]:
			db_proj = filter_db(self.db, cls, 6)
			prob[cls] = [{}, {}, {}, {}, {}, {}]
			for i in range(0, 6):
				for attr in attributes[i]:
					prob[cls][i][attr] = attr_prob(db_proj, attr, i)
					if prob[cls][i][attr] == 0:
						prob[cls][i][attr] = 0.001
		#print prob
		return prob

	# Compute all class probabilities: 
	# Output: 
	#	prob['acc'] = P('acc') 
	def compute_class_prob(self):
		prob = {}
		for cls in attributes[6]:
			prob[cls] = attr_prob(self.db, cls, 6)
		#print prob
		return prob

	# Predict the tuple's class
	def classify(self, tuple):
		prob = {}
		for cls in attributes[6]:
			prob_tmp = self.class_prob[cls]
			for i in range(0,6):
				attr_val = tuple[i]
				prob_tmp = prob_tmp + math.log(self.cond_prob[cls][i][attr_val])
			prob[cls] = prob_tmp
		#print prob
		return max(prob, key=prob.get)

		

# Parse data from file:
data_file = open("car.data", 'r')
db = [line.split(',') for line in data_file.read().splitlines()]

# Split database into training data and test data:
db_size = len(db)
db_training_size = int(db_size * 0.2)

random.shuffle(db)
db_train = db[db_training_size:]
db_test = db[:db_training_size]

# Naive Bayesian Classification:
nbc = nb_classifier(db_train)

accuracy_train = sum(bool(t[6] == nbc.classify(t)) for t in db_train) / float(len(db_train))
accuracy_test = sum(bool(t[6] == nbc.classify(t)) for t in db_test) / float(len(db_test))
print "Accuracy of NB Classifier:"
print "Training dataset:", accuracy_train
print "Test dataset:", accuracy_test

# Confusion Matrix:

conf_mat = {'unacc':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}, 
			'acc':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}, 
			'good':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}, 
			'vgood':{'unacc':0, 'acc':0, 'good':0, 'vgood':0}}

for t in db_test:
	actual_class = t[6]
	predicted_class = nbc.classify(t)
	conf_mat[actual_class][predicted_class] += 1

print conf_mat


