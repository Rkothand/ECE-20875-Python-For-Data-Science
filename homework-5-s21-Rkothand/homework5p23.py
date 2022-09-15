import numpy as np
from scipy.stats import norm

data = [3, -3, 3, 15, 15, -16, 14, 21, 30, -24, 32]

avg=np.mean(data)

sample_size = len(data)

stdev= 16.836

std_error = np.std(data, ddof=1)/(sample_size**0.5)

c=0.95

t_c= norm.t.ppf((1-(1-c)/2),df)

p= 2*norm.cdf(-abs(t_c))

conf= [avg-(t_c*(std_error/(sample_size**0.5))),avg+(t_c*(std_error/(sample_size**0.5)))]

print("the sample size is:", sample_size)
print("the sample mean is: ", avg)
print("the standard error is: ", std_error)
print("the t score value is: ", t_c)
print("the p value is:", p)
print("the confidence interval is: ", conf)
