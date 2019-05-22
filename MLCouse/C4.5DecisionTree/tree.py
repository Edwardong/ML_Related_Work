from __future__ import division
from math import log

class Tree:
    leaf = True
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None

def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    if (point.values[i] < tree.threshold):
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)

def most_likely_class(prediction):
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]

def accuracy(data, predictions):
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return float(correct) / total

def split_data(data, feature, threshold):
    left = []
    right = []
    # TODO: split data into left and right by given feature.
    # left should contain points whose values are less than threshold
    # right should contain points with values greater than or equal to threshold
    for point in data:
        if point.values[feature] < threshold:
            left.append(point)
        else:
            right.append(point)

    return (left, right)

def count_labels(data):
    counts = {}
    # TODO: counts should count the labels in data
    # e.g. counts = {'spam': 10, 'ham': 4}
    for point in data:
        if point.label in counts.keys():
            counts[point.label] += 1
        else:
            counts[point.label] = 1
    return counts

def counts_to_entropy(counts):
    entropy = 0.0
    # TODO: should convert a dictionary of counts into entropy
    total = sum(counts.itervalues())
    for key, value in counts.iteritems():
        entropy -= (value/total) * log(value/total, 2)
    return entropy
    
def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

def find_best_threshold(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    for point in data:
        left, right = split_data(data, feature, point.values[feature])
        curr = (get_entropy(left)*len(left) + get_entropy(right)*len(right))/len(data)
        gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = point.values[feature]
    return (best_gain, best_threshold)

def find_best_threshold_fast(data, feature):
    entropy = get_entropy(data)
    # TODO: Write a more efficient method to find the best threshold.
    """
    - Sort the dataset by the feature we are splitting on.
    - Go through the sorted data in order, moving data points from the right split to the left and
    - Keeping a rolling count of the probabilities of each label
    """
    data = sorted(data, key=lambda d: d.values[feature], reverse=True)
    #entropy_list = [] * len(data)

    # initially left = data[0], right = remaining

    left_counts = count_labels(data[:0])
    right_counts = count_labels(data[0:])
    left_sum = sum(left_counts.itervalues())
    right_sum = sum(right_counts.itervalues())
    # value * (log(value,2))
    left_vlogv2 = 0 # fill
    right_vlogv2 = 0 # fill
    for key, value in left_counts.iteritems():
        left_vlogv2 += value * log(value, 2)

    for key, value in right_counts.iteritems():
        right_vlogv2 += value * log(value, 2)
        
    left_entropy = counts_to_entropy(left_counts) # fill in
    right_entropy = counts_to_entropy(right_counts)# fill in

    best_gain = (left_entropy + right_entropy)/float(len(data))
    best_threshold = None

    for i in range(0, len(data)):
        # print(i)
        # update left&right entropy here
        # only 1 data point change
        # from left to right
        point = data[i]
        left_sum += 1
        right_sum -= 1        
        v = None

        # eliminate old terms
        if point.label in left_counts:
            v = left_counts[point.label]
            left_vlogv2 -= v * log(v, 2)
        else:
            v = 0

        v = right_counts[point.label]
        right_vlogv2 -= v * log(v, 2)

        # update new point label
        if point.label in left_counts:
            left_counts[point.label] += 1
        else:
            left_counts[point.label] = 1
        assert left_counts[point.label] >= 0
        right_counts[point.label] -= 1
        assert right_counts[point.label] >= 0

        # compute new vlogv2
        v = left_counts[point.label]
        left_vlogv2 += v * log(v, 2)
        
        v = right_counts[point.label]
        if v != 0:
            right_vlogv2 += v * log(v, 2)
        else:
            right_vlogv2 += 0

        # compute entropy
        left_entropy = -1.0/left_sum * (left_vlogv2 - log(left_sum, 2) * left_sum)# compute
        if right_sum == 0:
            right_entropy = 0
        else:
            right_entropy = -1.0/right_sum * (right_vlogv2 - log(right_sum, 2) * right_sum)# compute


        eps = 0.001
        # print(left_entropy, get_entropy(data[:i+1]))
        # assert abs(left_entropy - get_entropy(data[:i+1])) < eps
        # print("right", right_entropy, get_entropy(data[i+1:]))
        # assert abs(right_entropy - get_entropy(data[i+1:])) < eps
        length = len(data)
        curr = (left_entropy*(i+1) + right_entropy*(length-i-1))/float(len(data))
        gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = data[i].values[feature]
    '''
    for i in range(1, len(data)):
        left, right = data[:i], data[i:]
        curr = (get_entropy(left)*len(left) + get_entropy(right)*len(right))/len(data)
        gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = data[i-1].values[feature]
    '''
    return (best_gain, best_threshold)

def find_best_split(data):
    if len(data) < 2:
        return None, None
    best_feature = None
    best_threshold = None
    best_gain = 0
    # TODO: find the feature and threshold that maximize information gain.
    for feature in range(len(data[0].values)):
        gain, threshold = find_best_threshold_fast(data, feature)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold
    # print best_feature, best_threshold, best_gain
    return (best_feature, best_threshold)

def make_leaf(data):
    tree = Tree()   
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label])/len(data)
    tree.prediction = prediction
    return tree

pos = 0
def c45(data, max_levels):
    # TODO: Construct a decision tree with the data and return it.
    # Your algorithm should return a leaf if the maximum level depth is reached
    # or if there is no split that gains information, otherwise it should greedily
    # choose an feature and threshold to split on and recurse on both partitions
    # of the data.
    features = set()
    for point in data:
        for i in range(len(point.values)):
            # features.add(point.values[i])
            features.add(i)
    features = list(features)
    #print("numfeature=", len(features))
    pos = 0
    return helper(features, data, max_levels)
    '''
    for point in data:
        for i in range(len(point.values)):
            feature = point.values[i]
            gain,tpthreshold = find_best_threshold_fast(data,i)
            if gain == 0:
                return make_leaf(data)
            elif threshold > tpthreshold:
                threshold = tpthreshold
    '''
def helper(features, data, max_levels):
    if len(data) == 0 or max_levels <= 0: # or pos >= len(features):
        return make_leaf(data)
    best_feature = features[0]
    best_gain, best_threshold = find_best_threshold_fast(data, best_feature)
    for i in range(1, len(features)):
        feature = features[i]
        gain, threshold = find_best_threshold_fast(data, feature)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold
    gain = best_gain
    feature = best_feature
    threshold = best_threshold
    eps = 0.0001
    if abs(gain) < eps:
        return make_leaf(data)
    # threshold = None
    left, right = split_data(data, feature, threshold)
    tree = Tree();
    tree.leaf = False
    tree.threshold = threshold
    tree.left = helper(features, left, max_levels-1)
    tree.right = helper(features, right, max_levels-1)
    tree.feature = feature
    return tree
    


def submission(train, test):
    # TODO: Once your tests pass, make your submission as good as you can!
    tree = c45(train, max_levels=9)
    # print_tree(tree)
    predictions = []
    for point in test:
        predictions.append(predict(tree, point))
    return predictions

# This might be useful for debugging.
def print_tree(tree):
    if tree.leaf:
        print "Leaf", tree.prediction
    else:
        print "Branch", tree.feature, tree.threshold
        print_tree(tree.left)
        print_tree(tree.right)


