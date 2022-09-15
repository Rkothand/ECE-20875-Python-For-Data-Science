import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def main(data):
    # Since we want to use for-loop, change main to pass in dataframe instead of importing it here
    passed_50_data = data
   
    #Feature and target matrices
    X = passed_50_data[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]
    y = passed_50_data[['s']]
    
    #Training and testing split, with 25% of the data reserved as the test set
    X = X.to_numpy()
    y = y.to_numpy()
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    model_best = train_model(X_train, y_train)
    accscore = model_best.score(X_test, y_test)
    # TODO: decide whether we want to mess around with regularization or not
    #Define the range of lambda to test
    # lmbda=np.logspace(-1,2, num=51)

    # MODEL = []
    # ACC = []
    # for l in lmbda:
    #     #Train the regression model using a regularization parameter of l
    #     model = train_model(X_train,y_train,l)

    #     #Evaluate the MSE on the test set
    #     acc = error(X_test,y_test,model)

    #     #Store the model and mse in lists for further processing
    #     MODEL.append(model)
    #     ACC.append(acc)

    # # don't want to generate 93 plots
    # # plt.plot(lmbda,MSE) #fill in
    # # plt.legend(loc="upper left")
    # # plt.title("MSE vs. Lambda")	
    # # plt.xlabel('Regularization Parameter Lambda')
    # # plt.ylabel('Mean Squared Error')
    # # plt.show()
    # # plt.savefig('graphingp2')
    # #Find best value of lmbda in terms of MSE
    # ind = (ACC.index(max(ACC)))#fill in
    # [lmda_best,MSE_best,model_best] = [lmbda[ind],ACC[ind],MODEL[ind]]

    # print('Best C tested is ' + str(lmda_best) + ', which yields an accuracy score of ' + str(MSE_best))

    return model_best, accscore

#code from homework 7 that we reused


#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):
    mean = []
    std = []
    X = []
    for i in range(7):
        col = X_train[:, i]
        colmean = np.mean(col)
        colstd = np.std(col)
        col = [(x - colmean) / colstd for x in col]   # normalizes each column
        X.append(col)
        mean.append(colmean)
        std.append(colstd)
    X = np.array(X)     # convert nested list into numpy array
    X = np.transpose(X) # transpose array to get proper columns and rows
    return X, mean, std


#Function that normalizes testing set according to mean and std of training set
#Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
#column in training set: trn_std
#Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):
    X = []
    for i in range(7):
        col = X_test[:, i]
        col = [(x - trn_mean[i]) / trn_std[i] for x in col]   # normalizes each column
        X.append(col)
    X = np.array(X)
    X = np.transpose(X)
    return X


# Change this function to do logistic regression instead of ridge regression
def train_model(X,y):
    model= LogisticRegression()
    model.fit(X,np.ravel(y))
    return model


# Use accuracy score instead of MSE
def error(X,y,model):
    score = model.score(X, y)
    return score

if __name__ == '__main__':
    # import data
    data = pd.read_csv("behavior-performance.txt", sep = '\t', header=0)
    score_list = []
    for i in range(93):     # run main 93 times, each time passing in subset of dataframe containing only data for 1 video ID
        # bit of a hard-coded fix but there are no entries in the dataset for vidID 29, so skip it
        if i == 29:
            pass
        else:
            model_best, score = main(data.loc[data["VidID"] == i, :])
            print(f"Video ID {i} params: {model_best.coef_}")
            print(f"Video ID {i} score: {score * 100}")
            score_list.append(score * 100)
    print(f"Score minimum: {min(score_list)}")
    print(f"Score maximum: {max(score_list)}")
    print(f"Score average: {sum(score_list) / len(score_list)}")
    #We use the following functions to obtain the model parameters instead of model_best.get_params()


