import numpy as np
from scipy.stats import norm

myFile =open('engagement_0.txt')
data0 = myFile.readlines()
myFile.close()

data0 = [float(x) for x in data0]


myFile =open('engagement_1.txt')
data1 = myFile.readlines()
myFile.close()

data1 = [float(y) for y in data1]



sample_size0 = len(data0)
sample_size1 = len(data1)
avg0  = np.mean(data0)
avg1  = np.mean(data1)

error0=(sum((avg0-data0)**2))/(sample_size0-1)
error1=(sum((avg1-data1)**2))/(sample_size1-1)

sample_size2= sample_size0+sample_size1

avg2=(((avg0**2)/sample_size0)+((avg1**2)/sample_size1))**0.5

print("the sample size of engagement_0 is:", sample_size0)
print("the sample size of engagement_1 is:", sample_size1)
print("the sample size of engagement_2 is:", sample_size2)
print("the sample mean of the 2 sample test is : ", avg2)
print("the standard error0 is: ", error0)
print("the standard error1 is: ", error1)
'''
sample_size = len(data)

std_error = np.std(data, ddof=1)/(sample_size**0.5);

z_c =  (avg-0.75)/std_error

p = 2*norm.cdf(z_c)
##
significance1 = 0.1
significance2 = 0.05
significance3 = 0.01

(z_c2)=norm.ppf(0.05/2)

std_error2= (avg-0.75)/z_c2

if(significance1>p):
	print("the results are significant at", significance1)	
if(significance2>p):
	print("the results are significant at", significance2)	
if(significance3>p):
	print("the results are significant at", significance3)	

print("the sample size is:", sample_size)
print("the sample mean is: ", avg)
print("the standard error is: ", std_error)
print("the z score value is: ", z_c)
print("the p value is:", p)

print("the second error value is: ", std_error2)
'''
