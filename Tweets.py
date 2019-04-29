import random
import math
from csv import reader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


def importdata():
    with open('Gender_Classifier.csv',encoding="utf8", errors='ignore') as csvDataFile:
        csvReader = pd.read_csv(csvDataFile)
        df = pd.DataFrame(csvReader, columns=['_unit_id','_golden','_unit_state','_trusted_judgments','_last_judgment_at',
                                              'gender','gender:confidence','profile_yn','profile_yn:confidence','created',
                                              'description','fav_number','gender_gold','link_color','name','profile_yn_gold',
                                              'profileimage','retweet_count','sidebar_color','text','tweet_coord','tweet_count',
                                              'tweet_created','tweet_id','tweet_location','user_timezone'])
        return df

def Loading_Our_Dataset():
    data = open('Gender_Classifier.csv', encoding="utf8", errors="ignore")
    lines = reader(data)
    dataset = list(lines)
    return dataset

def Dataset_Cleansing(dataset):
    dataset.pop(0)
    for i in range(0, len(dataset)):
        dataset[i].pop(0) #Unit Id
        dataset[i].pop(0) #Golden
        dataset[i].pop(0) #Unit State
        dataset[i].pop(1) #last judment at
        dataset[i].pop(3) #profile_yn
        dataset[i].pop(4) #created
        dataset[i].pop(4) #description
        dataset[i].pop(5) #gender_gold
        dataset[i].pop(5) #link color
        dataset[i].pop(5) #name
        dataset[i].pop(5) #profile gold
        dataset[i].pop(5) #proflie image
        dataset[i].pop(6) #sidebar_color
        dataset[i].pop(6) #text
        dataset[i].pop(6) #tweet coord
        dataset[i].pop(7) #tweet created
        dataset[i].pop(7) #tweet id
        dataset[i].pop(7) #tweet location
        dataset[i].pop(7) #user timezone
    for i in range(0, len(dataset)):
        New_Thing = dataset[i].pop(1)
        dataset[i].append(New_Thing)
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i]) - 1):
            if dataset[i][j] == '':
                dataset[i][j] = '0'
    for i in range(0, len(dataset)):
        j = len(dataset[i]) - 1
        if dataset[i][j] != 'male' and dataset[i][j] != 'female' and dataset[i][j] != 'brand' and dataset[i][j] != 'unknown':
            dataset[i][j] = 'unknown'
    return dataset


def splitDataSet(csvReader):
    Y = csvReader.values[:, 5]
    for i in range(0, len(Y)):
        if Y[i] != 'male' and Y[i] != 'female' and Y[i] != 'unknown' and Y[i] != 'brand':
            Y[i] = 'unknown'
    X = csvReader[['_trusted_judgments', 'gender:confidence', 'profile_yn:confidence', 'retweet_count', 'tweet_count', 'fav_number']]
    X = X.fillna(X.mean())
    X = X.values[:, 0:5]
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.5, random_state= 100)
    return X, Y, X_train, X_test, Y_train, Y_test

def Splitting_Dataset_NoSK(dataset):
    Split_Dataset = list()
    Copied_Dataset = dataset
    Size = math.trunc(len(dataset) / 5)
    Folded_Set = list()
    i = 0
    while i < 5:
        while len(Folded_Set) < Size:
            index = random.randrange(0, len(Copied_Dataset))
            popped_value = Copied_Dataset.pop(index)
            Folded_Set.append(popped_value)
        Split_Dataset.append(Folded_Set)
        Folded_Set = list()
        i += 1
    return Split_Dataset



def train_with_gini_classifier(X_train, X_test, Y_train):
    gini_classifier = DecisionTreeClassifier(criterion="gini", random_state = 100, max_depth=200,
                                            min_samples_leaf=100, splitter="best", min_weight_fraction_leaf=0)
    gini_classifier.fit(X_train, Y_train)
    return gini_classifier


def train_with_entropy_classifier(X_train, X_test, Y_train):
    entropy_classifier = DecisionTreeClassifier(criterion="entropy", random_state= 100, max_depth=200,
                                                min_samples_leaf=100, splitter="best", min_weight_fraction_leaf=0)
    entropy_classifier.fit(X_train, Y_train)
    return entropy_classifier

def prediction(X_test, classifier):
    y_pred = classifier.predict(X_test)
    return y_pred

def calculate_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
           accuracy_score(y_test, y_pred)*100)

    print("Report : ",
          classification_report(y_test, y_pred))


#Everything Below here are Definitions that are used to determine Accuracy without SkLearn

def Gini_Evaluation_And_Clusters_NoSK(classes, index, value, dataset):
    Left_Classify = []
    Right_Classify = []
    for row in dataset:
        if row[index] < value:
            Left_Classify.append(row)
        else:
            Right_Classify.append(row)
    clusters = Left_Classify, Right_Classify
    Number_Of_Groups = 0.0
    for cluster in clusters:
        Number_Of_Groups += len(cluster)
    gini = 0.0
    for cluster in clusters:
        size = float(len(cluster))
        if size != 0:
            score = 0.0
            for class_val in classes:
                p = 0
                for row in cluster:
                    if row[len(row) - 1] == class_val:
                        p += 1
                p = p / size
                p = p * p
                score = score + p
            gini += (1.0 - score) * (size / Number_Of_Groups)
    return gini, clusters


def Terminal(group):
    Outcomes = []
    Unique_Outcomes = []
    for row in group:
        Outcomes.append(row[len(row) - 1])
    for items in Outcomes:
        if items not in Unique_Outcomes:
            Unique_Outcomes.append(items)
    return max(Unique_Outcomes, key=Outcomes.count)


def Tree_Builder(train, max_depth, min_size):
    root = Defining_Groups(train)
    Tree_Splitting(root, max_depth, min_size, 1)
    return root


def Defining_Groups(dataset):
    classifiers = []
    for row in dataset:
        if row[len(row) - 1] not in classifiers:
            classifiers.append(row[-1])
    index, value, score, initial_clusters = 999, 999, 999, None
    for i in range(0, len(dataset[0]) - 1):
        for row in dataset:
            gini, clusters = Gini_Evaluation_And_Clusters_NoSK(classifiers, i, row[i], dataset)
            if gini < score:
                index, value, score, initial_groups = i, row[i], gini, clusters
    return {'cluster': initial_groups, 'index': index, 'value': value}


def Tree_Splitting(node, max_depth, min_size, depth):
    left, right = node['cluster']
    del (node['cluster'])
    if not left or not right:
        node['left'] = node['right'] = Terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = Terminal(left), Terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = Terminal(left)
    else:
        node['left'] = Defining_Groups(left)
        Tree_Splitting(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = Terminal(right)
    else:
        node['right'] = Defining_Groups(right)
        Tree_Splitting(node['right'], max_depth, min_size, depth + 1)


def Gathering_Scores(dataset, algorithm, max_depth, min_size):
    Split_Dataset = Splitting_Dataset_NoSK(dataset)
    List_Of_Scores = list()
    for fold in Split_Dataset:
        Training_Set = list(Split_Dataset)
        Training_Set.remove(fold)
        Training_Set = sum(Training_Set, [])
        Testing_set = list()
        for row in fold:
            List_Of_Rows = list(row)
            Testing_set.append(List_Of_Rows)
            List_Of_Rows[len(List_Of_Rows) - 1] = None
        predicted = algorithm(Training_Set, Testing_set, max_depth, min_size)
        actual = list()
        for row in fold:
            actual += [row[len(row) - 1]]
        Percentage_Correct = Scoring(actual, predicted)
        List_Of_Scores.append(Percentage_Correct)
    return List_Of_Scores


def Scoring(actual, predicted):
    correct = 0
    answer = 0
    for i in range(0, len(actual)):
        if predicted[i] == actual[i]:
            correct += 1
    answer = correct / len(actual) * 100.0
    return answer


def Predictor(node, row):
    if row[node['index']] < node['value']:
        if type(node['left']) == dict:
            return Predictor(node['left'], row)
        else:
            return node['left']
    else:
        if type(node['right']) == dict:
            return Predictor(node['right'], row)
        else:
            return node['right']


def Decision_Tree_Algorithm(train, test, max_depth, min_size):
    tree = Tree_Builder(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = Predictor(tree, row)
        predictions.append(prediction)
    return (predictions)


def main():
    data = importdata()
    X, Y, X_train, X_test, Y_train, Y_test = splitDataSet(data)

    gini_classifier = train_with_gini_classifier(X_train, X_test, Y_train)
    entropy_classifier = train_with_entropy_classifier(X_train, X_test, Y_train)

    print("Gini Index Evaluation:")

    #Gini Classification
    Y_prediction_with_gini = prediction(X_test, gini_classifier)
    calculate_accuracy(Y_test, Y_prediction_with_gini)

    print("Entropy Evaluation:")
    # Entropy Classification
    Y_prediction_with_entropy = prediction(X_test, entropy_classifier)
    calculate_accuracy(Y_test, Y_prediction_with_entropy)

    # End of the SKLearn Part

    # This is an Implementation without SKLearn
    # You should comment this out if you only want to look at the SK Learn part, since this takes about 3 hours to run
    # on my machine.
    dataset = Loading_Our_Dataset()
    dataset = Dataset_Cleansing(dataset)
    scores = Gathering_Scores(dataset, Decision_Tree_Algorithm, 5, 10)
    print('Score for 1 Fold: ', scores[0], '%')
    print('Score for second Fold: ', scores[1], '%')
    print('Score for Third Fold: ', scores[2], '%')
    print('Score for Fourth Fold: ', scores[3], '%')
    print('Score for Last Fold: ', scores[4], '%')
    print('Average for all Folds: ', (sum(scores) / len(scores)), '%')

if __name__ == "__main__":
    main()
