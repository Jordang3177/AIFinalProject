import random
import math
from csv import reader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


# This is used to import the data in a format that is utilized for SKLearn, with Panda.
# I established each of the columns so that they could referenced easily.
# This link was used to help with the importing of this data and using the panda data frames:
# https://towardsdatascience.com/a-quick-introduction-to-the-pandas-python-library-f1b678f34673
def importdata():
    with open('Gender_Classifier.csv', encoding="utf8", errors='ignore') as csvDataFile:
        csvreader = pd.read_csv(csvDataFile)
        df = pd.DataFrame(csvreader, columns=['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
                                              '_last_judgment_at', 'gender', 'gender:confidence', 'profile_yn',
                                              'profile_yn:confidence', 'created', 'description', 'fav_number',
                                              'gender_gold', 'link_color', 'name', 'profile_yn_gold', 'profileimage',
                                              'retweet_count', 'sidebar_color', 'text', 'tweet_coord', 'tweet_count',
                                              'tweet_created', 'tweet_id', 'tweet_location', 'user_timezone'])
        return df


# This is meant to load in the dataset as a list rather than a dataset to be used below
def loading_our_dataset():
    data = open('Gender_Classifier.csv', encoding="utf8", errors="ignore")
    lines = reader(data)
    dataset = list(lines)
    return dataset


# This is utilized to cleanse the dataset of any abnormalities. As well as popping all the data
# That I don't want to use in the dataset.
def dataset_cleansing(dataset):
    # I had to pop the names of the columns out for an easier access of the numbers in the dataset
    dataset.pop(0)
    # This is where i'm popping all of the data that I don't need
    # This would need to be completely changed if you want to access different data because it's popping one at a time
    for i in range(0, len(dataset)):
        dataset[i].pop(0)  # Unit Id
        dataset[i].pop(0)  # Golden
        dataset[i].pop(0)  # Unit State
        dataset[i].pop(1)  # last judgment at
        dataset[i].pop(3)  # profile_yn
        dataset[i].pop(4)  # created
        dataset[i].pop(4)  # description
        dataset[i].pop(5)  # gender_gold
        dataset[i].pop(5)  # link color
        dataset[i].pop(5)  # name
        dataset[i].pop(5)  # profile gold
        dataset[i].pop(5)  # profile image
        dataset[i].pop(6)  # sidebar_color
        dataset[i].pop(6)  # text
        dataset[i].pop(6)  # tweet coord
        dataset[i].pop(7)  # tweet created
        dataset[i].pop(7)  # tweet id
        dataset[i].pop(7)  # tweet location
        dataset[i].pop(7)  # user timezone
    # I wanted to move the classifier to the end of the dataset to make it easier to iterate over things and
    # easier to access the classification
    for i in range(0, len(dataset)):
        new_thing = dataset[i].pop(1)
        dataset[i].append(new_thing)
    # Replacing all the null values with zero.
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i]) - 1):
            if dataset[i][j] == '':
                dataset[i][j] = '0'
    # If there are any null values for the classifier I will put them as unknown.
    for i in range(0, len(dataset)):
        j = len(dataset[i]) - 1
        if dataset[i][j] != 'male' and dataset[i][j] != 'female' and \
                dataset[i][j] != 'brand' and dataset[i][j] != 'unknown':
            dataset[i][j] = 'unknown'
    return dataset


# This is used to split the dataset for the Sklearn algorithm to be used
def split_dataset(csvreader):
    # y will be used as the classifier set.
    y = csvreader.values[:, 5]
    # Cleansing the dataset to exclude null values
    for i in range(0, len(y)):
        if y[i] != 'male' and y[i] != 'female' and y[i] != 'unknown' and y[i] != 'brand':
            y[i] = 'unknown'
    # Telling the x values which columns to take from.
    x = csvreader[['_trusted_judgments', 'gender:confidence', 'profile_yn:confidence',
                   'retweet_count', 'tweet_count', 'fav_number']]
    # Filling the values with the mean of the dataset.
    x = x.fillna(x.mean())
    # x values here will be those listed above.
    x = x.values[:, 0:5]
    # This is where we start using Sklearn algorithms and here it's to make a training set split on x and y,
    # and I decided to do 50% as it seems to yield one of the best results.
    x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size=0.5, random_state=100)
    return x, y, x_train, x_test, y_train, y_test


# This is where we will do a Decision Tree Classifier with the Gini index in mind,
# I also made the max depths insanely high because I wanted this to be as accurate as it could be.
# In order to help my understanding of how to use SKlearn I utilized this source:
# https://www.geeksforgeeks.org/decision-tree-implementation-python/
def train_with_gini_classifier(x_train, y_train):
    gini_classifier = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=200,
                                             min_samples_leaf=100, splitter="best", min_weight_fraction_leaf=0)
    gini_classifier.fit(x_train, y_train)
    return gini_classifier


# This is the same as the one above except that we will be using Entropy for this instead of using Gini.
def train_with_entropy_classifier(x_train, y_train):
    entropy_classifier = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=200,
                                                min_samples_leaf=100, splitter="best", min_weight_fraction_leaf=0)
    entropy_classifier.fit(x_train, y_train)
    return entropy_classifier


# This is used to be a prediction metric for the actual against the test
def prediction(x_test, classifier):
    y_pred = classifier.predict(x_test)
    return y_pred


# This is where we will look at the accuracy of the testing done above.
def calculate_accuracy(y_test, y_pred):

    # In order for me to understand what exactly the confusion matrix meant I looked up at this website:
    # https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)
    # Source:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    print("Report : ",
          classification_report(y_test, y_pred))


# Everything Below here are Definitions that are used to determine Accuracy without SkLearn

# In this part I tried to make a decision tree from scratch and below are the resources I used to accomplish this:
# This one was extremely helpful in understanding how decision trees are made in both R and Python
# as well as the pruning of the tree, which is something I would like to implement next in my code.
# https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
# http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
# http://openbookproject.net/thinkcs/python/english2e/ch21.html
# https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/
# This next one does have a python implementation for what I have here,
# but I tried to only use the words that he stated as a guide rather than looking directly looking at the code unless
# I was completely stuck. So this link was only utilized to help guide me on a frame of mind and
# not just copy pasting code:
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/


# This is where we start splitting the data for the algorithms later.
# As you can see it will be based on a 5 decision that I hard code in, this can be changed easily by just
# replacing both the instances of 5 with whichever number you would like to use.
def splitting_dataset_nosk(dataset):
    # This is where I'll be storing the split part of the dataset.
    split_datasets = list()
    # This is used to determine where I need to break the while loop later
    # for later use if you want to make the list of scores longer you can change this five to be any number
    size = math.trunc(len(dataset) / 5)
    # This is so that I have have something to append with.
    folded_set = list()
    i = 0
    # Also if you want to change the length of the list of scores you need to change this number as well.
    while i < 5:
        while len(folded_set) < size:
            # Using a random Range to just pick out a value and pop it into the folded set.
            index = random.randrange(0, len(dataset))
            # Stores the popped value
            popped_value = dataset.pop(index)
            # Appends the popped value in order to extend the
            # size of the folded list to make the while loop stop eventually.
            folded_set.append(popped_value)
        # Starts to form the split dataset
        split_datasets.append(folded_set)
        # resets the folded set to do the while loop again.
        folded_set = list()
        i += 1
    return split_datasets


# We are defining our Gini index here as well as the clusters for the groups to be used later.
# I need to come back and improve upon this later as it seems this is where almost all of the runtime is coming from.
# This source helped with Gini:
# https://www.geeksforgeeks.org/decision-tree-introduction-example/
def gini_evaluation_and_clusters_nosk(classes, index, value, dataset):
    # This is to either classify it above or below the line for the classes that are thrown in here.
    left_classify = list()
    right_classify = list()
    # This is where we define where each value fits.
    for row in dataset:
        if row[index] < value:
            left_classify.append(row)
        else:
            right_classify.append(row)
    # We need to make them all clustered together into something to classify them on later.
    clusters = left_classify, right_classify
    number_of_groups = 0
    # We need to figure out just how many clusters are in the
    # dataset in order to find our gini index based on the algorithm.
    for clump in clusters:
        number_of_groups += len(clump)
    gini = 0
    # This is where the main part of the Gini algorithm takes place
    for clump in clusters:
        # First we need to find the size of this clump.
        size = len(clump)
        tally = 0
        # Then as long as it isn't zero we can evaluate on it.
        if size != 0:
            # Now we need to figure out which of the classifications
            # it is and if it's the one we need then we add one to the p-value.
            for classification in classes:
                # Resetting the P value each time for each classification
                p = 0
                for row in clump:
                    if row[len(row) - 1] == classification:
                        p += 1
                # This is just the algorithm for finding the gini value.
                # Having to find how many of each classification there is in each clump
                # and adding that to the total amount which I made to be tally, and you do 1 - the total tally so far,
                # and then multiply it by the size of the clump and divide it by the length of all the clumps
                p = p / size
                p = p * p
                tally = tally + p
            gini += (1 - tally) * (size / number_of_groups)
    return gini, clusters


# This was used to help with getting terminal cases for the tree later.
def terminal(group):
    # Storing values in order to keep track which classifications are found.
    outcomes = list()
    unique_outcomes = list()
    for row in group:
        outcomes.append(row[len(row) - 1])
    for items in outcomes:
        if items not in unique_outcomes:
            unique_outcomes.append(items)
    return max(unique_outcomes, key=outcomes.count)


# This is the making of the tree, you can see the Main tree building part at Tree Splitting and
# the way the nodes are made in defining_groups
def tree_builder(train, max_depth, min_size):
    root = defining_groups(train)
    tree_splitting(root, max_depth, min_size, 1)
    return root


# This is where we need to make the nodes for the tree.
def defining_groups(dataset):
    classifiers = list()
    # This is where we are going to be setting all the classifiers that we have in the data set
    # For ours it is going to be male, female, brand, and unknown.
    # I wanted to keep unknown as I believe that it can be helpful to determine if it doesn't know
    # which one to group them in it will put it as unknown.
    for row in dataset:
        if row[len(row) - 1] not in classifiers:
            classifiers.append(row[-1])
    # This is just initialization of some values to be used to categorize the data.
    index, value, score, initial_clusters = 400, 400, 400, None
    # we will be iterating over each row in the list as well as keeping track of which column that we are in.
    for column in range(0, len(dataset[0]) - 1):
        for row in dataset:
            # We need to find the gini value and the clusters that we will be using for later.
            gini, clusters = gini_evaluation_and_clusters_nosk(classifiers, column, row[column], dataset)
            # and if the gini is less than current value for score then we need to change all of the values.
            if gini < score:
                index, value, score, initial_clusters = column, row[column], gini, clusters
    # And once we are done evaluating the dataset we can return this node.
    return {'cluster': initial_clusters, 'index': index, 'value': value}


# This is the main part of making the tree. in which we are only going so far down the tree as well
# and if we hit that point then we make a terminal node.
def tree_splitting(node, max_depth, min_size, depth):
    # Need to first define the left and right values as nodes which have the cluster
    left, right = node['cluster']
    # if the left or the right are empty then we need to make them terminal nodes with all of each others values
    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = terminal(left + right)
        return
    # if we have hit the max depth then we need to make terminal nodes for both sides.
    if depth >= max_depth:
        node['left'], node['right'] = terminal(left), terminal(right)
        return
    # if the left node is lower than the minimum size then we will make it terminal.
    if len(left) <= min_size:
        node['left'] = terminal(left)
    else:
        # otherwise we have to figure out which groups to give it.
        node['left'] = defining_groups(left)
        # as well as going deeper into the tree.
        tree_splitting(node['left'], max_depth, min_size, depth + 1)
    # Same as the right side for the above left side.
    if len(right) <= min_size:
        node['right'] = terminal(right)
    else:
        node['right'] = defining_groups(right)
        tree_splitting(node['right'], max_depth, min_size, depth + 1)


# This is where we will be gathering the accuracy of the data that we have.
def gathering_scores(dataset, algorithm, max_depth, min_size):
    # First we need the split dataset
    folded_dataset = splitting_dataset_nosk(dataset)
    # And just making an empty list to score the scores in for later use.
    list_of_scores = list()
    # Then for each fold that is in the dataset we will be iterating on.
    # Where we will be looking at all the rows in each fold and then utilizing this to
    # see if the algorithm is able to correctly classify and if so just how many times it is able to do this.
    for fold in folded_dataset:
        training_set = list(folded_dataset)
        training_set.remove(fold)
        training_set = sum(training_set, [])
        testing_set = list()
        # This is where we are gathering the testing set to be used in the decision tree algorithm
        for row in fold:
            list_of_rows = list(row)
            testing_set.append(list_of_rows)
            list_of_rows[len(list_of_rows) - 1] = None
        # finding the predicted values from the algorithm given, and here it will be the decision tree.
        predicted = algorithm(training_set, testing_set, max_depth, min_size)
        # Now we need to gather the actual values from the dataset.
        actual = list()
        for row in fold:
            actual += [row[len(row) - 1]]
        # Once we have both the actual and the predicted values we can now check to see how good our predictions were.
        correct = 0
        for i in range(0, len(actual)):
            if predicted[i] == actual[i]:
                correct += 1
        percentage_correct = correct / len(actual) * 100
        # and store them in a list to be displayed at the end.
        list_of_scores.append(percentage_correct)
    return list_of_scores


# This is evaluating which node to return based on the given nodes value.
def predictor(node, row):
    if row[node['index']] < node['value']:
        if type(node['left']) == dict:
            return predictor(node['left'], row)
        else:
            return node['left']
    else:
        if type(node['right']) == dict:
            return predictor(node['right'], row)
        else:
            return node['right']


# This is the call for the decision tree to be made and use the predictor algorithm in order to return the predictions.
def decision_tree_algorithm(train, test, max_depth, min_size):
    tree = tree_builder(train, max_depth, min_size)
    predictions = list()
    for row in test:
        predictions.append(predictor(tree, row))
    return predictions


def main():
    # Beginning of the SKlearn part.
    data = importdata()
    x, y, x_train, x_test, y_train, y_test = split_dataset(data)

    gini_classifier = train_with_gini_classifier(x_train, y_train)
    entropy_classifier = train_with_entropy_classifier(x_train, y_train)

    print("Gini Index Evaluation:")

    # Gini Classification
    y_prediction_with_gini = prediction(x_test, gini_classifier)
    calculate_accuracy(y_test, y_prediction_with_gini)

    print("Entropy Evaluation:")
    # Entropy Classification
    y_prediction_with_entropy = prediction(x_test, entropy_classifier)
    calculate_accuracy(y_test, y_prediction_with_entropy)

    # End of the SKLearn Part

    # This is an Implementation without SKLearn
    # You should comment this out if you only want to look at the SK Learn part, since this takes about 3 hours to run
    # on my machine.
    dataset = loading_our_dataset()
    dataset = dataset_cleansing(dataset)
    scores = gathering_scores(dataset, decision_tree_algorithm, 5, 10)
    print('Score for 1 Fold: ', scores[0], '%')
    print('Score for second Fold: ', scores[1], '%')
    print('Score for Third Fold: ', scores[2], '%')
    print('Score for Fourth Fold: ', scores[3], '%')
    print('Score for Last Fold: ', scores[4], '%')
    print('Average for all Folds: ', (sum(scores) / len(scores)), '%')

    # Ending of the Decision tree without SkLearn.


if __name__ == "__main__":
    main()
