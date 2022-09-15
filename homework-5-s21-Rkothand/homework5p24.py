import numpy as np
from scipy.stats import norm

data = [3, -3, 3, 15, 15, -16, 14, 21, 30, -24, 32]

avg=np.mean(data)

sample_size = len(data)

std_error = np.std(data, ddof=1)/(sample_size**0.5)

c=0.95

z_c = norm.ppf((1-(1-c)/2))

p= 2*norm.cdf(-abs(z_c))

conf= [avg-(z_c*(std_error)),avg+(z_c*(std_error))]

print("the sample size is:", sample_size)
print("the sample mean is: ", avg)
print("the standard error is: ", std_error)
print("the z score value is: ", z_c)
print("the p value is:", p)
print("the confidence interval is: ", conf)
