import numpy as np
from scipy.stats import norm

myFile =open('engagement_0.txt')
data = myFile.readlines()
myFile.close()

data = [float(x) for x in data]



avg  = np.mean(data)

sample_size = len(data)

std_error = np.std(data, ddof=1)/(sample_size**0.5);

z_c =  (avg-0.75)/std_error

p = 2*norm.cdf(z_c)

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
