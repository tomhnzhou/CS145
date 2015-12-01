import math
import pydot
import random

attributes = ['buying','maintenance','doors','persons','luggage_boot','safety']
all_attributes = ['buying','maintenance','doors','persons','luggage_boot','safety']
all_quality = ['unacc', 'acc', 'good', 'vgood']
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
        tree = {best[0]:{}}
        subset0 = best[1][0]
        subset1 = best[1][1]
        word1 = ''
        word2 = ''
        
        subtree = create_decision_tree(
        	get_examples(db, best[0], subset0),
            [attr for attr in attributes if attr != best]
            )

        for i in range(0, len(subset0)):
            if i == len(subset0) - 1:
                word1 += subset0[i]
            else:
                word1 += subset0[i] + '&'

        tree[best[0]][word1] = subtree

        subtree = create_decision_tree(
            get_examples(db, best[0], subset1),
            [attr for attr in attributes if attr != best]
            )

        for i in range(0, len(subset1)):
            if i == len(subset1) - 1:
                word2 += subset1[i]
            else:
                word2 += subset1[i] + '&'
        
        tree[best[0]][word2] = subtree

    return tree

def get_examples(db, attr, subset):
	"""Get the subset database for certain attribute with certain subset"""

	attr_num = attr_dict[attr]
	db_subset = []

	for record in db:
		if record[attr_num] in subset:
			db_subset.append(record) 

	return db_subset

def choose_attribute(db, attributes):
    """Find the best attribute in the given database"""
    minimum = ('', (), 2)
    result = ('', ())

    for attr in attributes:
        temp_gini = gini(db, attr)

        if temp_gini[1] < minimum[2]:
            minimum = (attr, temp_gini[0], temp_gini[1])

    result = (minimum[0], minimum[1])
    
    return result

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

def gini_helper(db):
    """Helper function of Gini index function"""
    val_freq = {}
    gini = 0

    for record in db:
        target_attr = record[6]
        if (val_freq.has_key(target_attr)):
            val_freq[target_attr] += 1.0
        else:
            val_freq[target_attr] = 1.0

    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        gini += -math.pow(val_prob, 2) 

    return 1+gini

def gini(db, attr):
    """Calculate Gini index with the given database and attribute"""

    vals = []
    attr_num = attr_dict[attr]
    split_results = []
    min_gini = ((), 2)

    # get all values in given attribute
    for record in db:
        if record[attr_num] not in vals:
            vals.append(record[attr_num])
    
    # get all possible binary split results
    for i in range(0, len(vals) - 1):
        temple_list = []
        for j in range(i, len(vals) - 1):
            temple_list.append(vals[j])
            item = (temple_list[:], [val for val in vals if (val not in temple_list)])
            split_results.append(item)

    # Check all possible splitting results
    for result in split_results:

        subset0_freq = 0
        subset1_freq = 0
        subset0 = result[0]
        subset0_db = []
        subset1_db = []

        #Calculate the frequency of each subset in every result
        for record in db:
            if record[attr_num] in subset0:
                subset0_freq += 1.0
                subset0_db.append(record)
            else:
                subset1_freq += 1.0
                subset1_db.append(record)

        # Calculate Gini
        temp_gini = (subset0_freq/len(db))*gini_helper(subset0_db) + (subset1_freq/len(db))*gini_helper(subset1_db)

        # Find the result with minimum Gini value
        if temp_gini < min_gini[1]:
            min_gini = (result, temp_gini)

    return min_gini

def walk_dictionaryv2(graph, dictionary, parent_node=None):
    """Recursive plotting function for the decision tree stored as a dictionary"""

    for k in dictionary.keys():

        if parent_node is not None:
            from_name = parent_node.get_name().replace("\"", "") + '_' + str(k)
            from_label = str(k)

            node_from = pydot.Node(from_name, label = from_label)
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
    graph.write_png('gini.png')

def accuracy_helper(record, tree, attributes):
    new_attributes = attributes[:]

    for attr in attributes:
        for k in tree.keys():
            if attr in k:
                sub_tree = tree[k]

                #Test if reach leaf node
                if sub_tree in all_quality:
                    return sub_tree

                if attr in all_attributes:
                    # new_attributes.remove(attr)
                    attr_num = attr_dict[attr]
                    new_attributes.insert(0, record[attr_num])

                    return accuracy_helper(record, sub_tree, new_attributes)
                else:
                    new_attributes.pop(0)
                    return accuracy_helper(record, sub_tree, new_attributes)


def accuracy(db, tree):
    right_num = 0

    for record in db:
        guess_result = accuracy_helper(record, tree, attributes)
        if guess_result == record[6]:
            right_num += 1.0

    return right_num/(len(db))

if __name__ == "__main__":

	# Parse training and testing data from file:
    train_file = open("train.data", 'r')
    db_train = [line.split(',') for line in train_file.read().splitlines()]
    test_file = open("test.data", 'r')
    db_test = [line.split(',') for line in test_file.read().splitlines()]

    tree = create_decision_tree(db_train, attributes)
    print accuracy(db_test, tree)
    plot_tree(tree)