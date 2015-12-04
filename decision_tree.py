import math
import pydot
import random

attributes = ['buying','maintenance','doors','persons','luggage_boot','safety']
attr_dict = {'buying':0, 'maintenance':1, 'doors':2, 'persons':3, 'luggage_boot':4, 'safety':5}
target_attr = 'quality'

def create_decision_tree(db, attributes):
    """Returns a new decision tree based on the given training database"""
    vals = [record[6] for record in db]
    default = majority_value(db)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not db or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(db, attributes)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(db, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
            	get_examples(db, best, val),
                [attr for attr in attributes if attr != best]
                )

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree

def get_examples(db, attr, val):
	"""Get the subset database for certain attribute with certain value"""
	attr_num = attr_dict[attr]
	db_subset = []

	for record in db:
		if record[attr_num] == val:
			db_subset.append(record) 

	return db_subset

def get_values(db, attr):
	"""Get all possible values of certain attribute in the database"""
	attr_num = attr_dict[attr]
	vals = []

	for record in db:
		if record[attr_num] not in vals:
			vals.append(record[attr_num])

	return vals

def choose_attribute(db, attributes):
	"""Find the best attribute in the given database"""
	maximum = ('', 0)

	for attr in attributes:
		gain = info_gain(db, attr)

		if gain > maximum[1]:
			maximum = (attr, gain)

	return maximum[0]

def majority_value(db):
	"""Find the majority value of target_attr in database"""
	temp_map = {}
	maximum = ('', 0)

	for record in db:
		val = record[6]

		if val in temp_map:
			temp_map[val] += 1
		else:
			temp_map[val] = 1

		if temp_map[val] > maximum[1]:
			maximum = (val, temp_map[val])

	return maximum[0]

def entropy(db):
    """Calculates the entropy of the given data set for the target attribute."""
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in db:
        if (val_freq.has_key(record[6])):
            val_freq[record[6]] += 1.0
        else:
            val_freq[record[6]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(db)) * math.log(freq/len(db), 2) 
        
    return data_entropy

def info_gain(db, attr):
    """Calculates the information gain on the chosen attribute (attr)."""
    val_freq = {}
    subset_entropy = 0.0
    attr_num = attr_dict[attr]

    # Calculate the frequency of each of the values in the target attribute
    for record in db:
        if (val_freq.has_key(record[attr_num])):
            val_freq[record[attr_num]] += 1.0
        else:
            val_freq[record[attr_num]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        db_subset = [record for record in db if record[attr_num] == val]
        subset_entropy += val_prob * entropy(db_subset)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(db) - subset_entropy)

def walk_dictionaryv2(graph, dictionary, parent_node=None):
    """Recursive plotting function for the decision tree stored as a dictionary"""

    for k in dictionary.keys():

        if parent_node is not None:
            from_name = parent_node.get_name().replace("\"", "") + '_' + str(k)
            from_label = str(k)

            if k in attributes:
                node_from = pydot.Node(from_name, label = from_label)
            else:
                node_from = pydot.Node(from_name, label = from_label, shape='plaintext')

            graph.add_node(node_from)

            graph.add_edge( pydot.Edge(parent_node, node_from) )

            # if interim node
            if isinstance(dictionary[k], dict):
                walk_dictionaryv2(graph, dictionary[k], node_from)

            # if leaf node
            else: 
            	num = random.random()

                to_name = str(k) + str(num) + str(dictionary[k]) # unique name
                to_label = str(dictionary[k])

                node_to = pydot.Node(to_name, label=to_label, shape='box')
                graph.add_node(node_to)
                graph.add_edge(pydot.Edge(node_from, node_to))

        else:

            from_name =  str(k)
            from_label = str(k)

            node_from = pydot.Node(from_name, label=from_label)
            walk_dictionaryv2(graph, dictionary[k], node_from)


def plot_tree(tree):

   # first you create a new graph, you do that with pydot.Dot()
    graph = pydot.Dot(graph_type='graph')
    walk_dictionaryv2(graph, tree)
    graph.write_png('tree.png')

if __name__ == "__main__":
	# Parse training and testing data from file:
	train_file = open("train.data", 'r')
	db_train = [line.split(',') for line in train_file.read().splitlines()]
	tree = create_decision_tree(db_train, attributes)
	plot_tree(tree)
