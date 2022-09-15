import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from helper import getData
data = getData('distC.csv')

stats.probplot(data, dist = 'rayleigh', plot=plt)
plt.show() # modify this to write the plot to a file instead

plt.savefig('graphC')
