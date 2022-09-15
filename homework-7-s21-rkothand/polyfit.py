import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
	paramFits = []
	file1 = open(datapath,'r')
	a = file1.readlines()
	x = []
	y = []
	for i in a:
		b=i.split()
		x.append(float(b[0]))
		y.append(float(b[1]))
	file1.close()
	for n in degrees:
		f1=feature_matrix(x,n)
		f2=least_squares(f1,y)
		paramFits.append(f2)		
    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in poly.txt.
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    #for the model parameters in each case. Append the result to paramFits each time.
	plt.scatter(x,y,label="original data")	
	x.sort()
	for k in range(1,6):	
		X=feature_matrix(x,k)
		plt.plot(x,np.dot(X,paramFits[k-1]),label = k)
	#B=least_squares(X,y)	
	plt.legend(loc="upper left")	
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show() # modify this to write the plot to a file instead
	plt.savefig('graph')
	#x2fm=feature_matrix(x,3)
	#x2ls=least_squares(x2fm,y)
	#print(x2ls)
	return paramFits
	

#Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and d as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):
	X=[[x[i]**j for j in range(d,-1,-1)] for i in range(len(x))]	
	
    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^d, x^(d-1), ..., x^0.
	return X


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
	X = np.array(X)
	y = np.array(y)
	B = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))

    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.

	return B

if __name__ == '__main__':
    datapath = 'poly.txt'
    degrees = [1,2,3,4,5]

    paramFits = main(datapath, degrees)
    print(paramFits)
	
