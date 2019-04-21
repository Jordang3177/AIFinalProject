import pandas as pd
import numpy as np
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
        #to_num_features = ['_trusted_judgments', 'gender:confidence', 'profile_yn:confidence', 'fav_number', 'retweet_count', 'tweet_count']
        #df = df[to_num_features].astype(float)

        # Printing the dataset shape
        print("Gender Data Length: ", len(df))
        print("Gender Data Dimensions: ", df.shape)

        # Printing the dataset obseravtions
        print("Gender Data First 5 Rows: ", df.head())
        return df

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def splitDataSet(csvReader):
    Y = csvReader.values[:, 5]
    for i in range(0, len(Y)):
        if Y[i] != 'male' and Y[i] != 'female' and Y[i] != 'unknown' and Y[i] != 'brand':
            Y[i] = 'unknown'
    X = csvReader[['_trusted_judgments', 'gender:confidence', 'profile_yn:confidence', 'retweet_count']]
    X = X.fillna(X.mean())
    X = X.values[:, 0:3]
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.3, random_state= 100)
    return X, Y, X_train, X_test, Y_train, Y_test

def train_with_gini_classifier(X_train, X_test, Y_train):
    gini_classifier = DecisionTreeClassifier(criterion="gini", random_state = 100, max_depth=3,
                                            min_samples_leaf=5, splitter="best", min_weight_fraction_leaf=0)
    gini_classifier.fit(X_train, Y_train)
    return gini_classifier

def train_with_entropy_classifier(X_train, X_test, Y_train):
    entropy_classifier = DecisionTreeClassifier(criterion="entropy", random_state= 100, max_depth=3,
                                                min_samples_leaf=5, splitter="best", min_weight_fraction_leaf=0)
    entropy_classifier.fit(X_train, Y_train)
    return entropy_classifier

def prediction(X_test, classifier):
    y_pred = classifier.predict(X_test)
    print("Prediction Values: ")
    print(y_pred)
    return y_pred

def calculate_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
           accuracy_score(y_test, y_pred)*100)

    print("Report : ",
          classification_report(y_test, y_pred))
def main():
    data = importdata()
    #data = clean_dataset(data)
    X, Y, X_train, X_test, Y_train, Y_test = splitDataSet(data)
    gini_classifier = train_with_gini_classifier(X_train, X_test, Y_train)
    entropy_classifier = train_with_entropy_classifier(X_train, X_test, Y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    Y_prediction_with_gini = prediction(X_test, gini_classifier)
    calculate_accuracy(Y_test, Y_prediction_with_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    Y_prediction_with_entropy = prediction(X_test, entropy_classifier)
    calculate_accuracy(Y_test, Y_prediction_with_entropy)

# Calling main function
if __name__ == "__main__":
    main()