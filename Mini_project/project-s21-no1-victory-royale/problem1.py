import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__' :
    # values = np.genfromtxt("behavior-performance.txt", delimiter = '\t', skip_header = 1, usecols = )
    data = pd.read_csv("behavior-performance.txt", sep = '\t', header=0)
    #print(data)
    print(f'Data shape: {data.shape}')

    ## Problem 1.
    # Want to use Naive Bayes with gaussian
    # Each class is a userID
    # Each feature is a video watching behavior parameter
    
    # keep students who have at least 5x fracComp = 1
    # define completed video as fracComp > 0.9
    #print(data.loc[data["fracComp"] > 0.9, :])
    num_completed = data.loc[data["fracComp"] > 0.9, :].groupby("userID")["fracComp"].count()
    userid_comp5 = list(num_completed[num_completed >= 5].index.values)     # list of userID's who completed 5 videos
    print(data.loc[data["userID"] == userid_comp5[0]])
    print(num_completed[:5])
    # 7 feature X
    X = data[data["userID"].isin(userid_comp5)].loc[:, ["fracSpent", "fracComp", "fracPaused", "numPauses", "avgPBR", "numRWs", "numFFs"]].to_numpy()
    print(f'Number of students fracComp=1 is >5: {len(userid_comp5)}')
    print(f'Shape of data for those students: {X.shape}')
    # TODO look into replacing empty cells (Na) to 0 or 1 or something
        # nvm the dataset doesnt appear to have any empty cells
    y = data[data["userID"].isin(userid_comp5)].loc[:, "userID"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print(y_pred[:5])
    
    print("Gaussian Naive Bayes model accuracy(in %):",	metrics.accuracy_score(y_test, y_pred)*100)
    # model accuracy according to this is only about 2%
    # could it be something to do with how the y values are strings? or just that we have so many classes that it's difficult to predict using Naive Bayes?

    # Maybe try k-nearest neighbors? since naive bayes is parametric, have to assume the data is distributed normally,
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, y_train)
    y_predkn = neigh.predict(X_test)
    print(y_predkn[:5])
    print("K Neighbors model accuracy(in %):", metrics.accuracy_score(y_test, y_predkn)*100)
    # k-nearest neighbors has double the accuracy, around 4%.
    # low accuracy could be because there are so many features, and so many classes, that choosing the closest class is difficult

    # try again with k = 3 instead of k = 5
    neigh2 = KNeighborsClassifier(n_neighbors = 3)
    neigh2.fit(X_train, y_train)
    y_predkn2 = neigh2.predict(X_test)
    print("K Neighbors model 2 accuracy(in %):", metrics.accuracy_score(y_test, y_predkn2)*100)
    # accuracy is even lower this time, only 3.7%