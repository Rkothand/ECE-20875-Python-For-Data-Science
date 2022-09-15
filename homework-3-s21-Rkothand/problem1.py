import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(hist):
	total=sum(hist)
	hist=hist/total
	return hist




def compute_j(histo, width):
	histo2=norm_histogram(histo)
	sump = [] 
	for i in histo2:
		p2=i**2
		sump.append(p2)
	total = sum(sump)
	total = float(total)
	m=sum(histo)
	m = float(m)
	
	jval = 2/((m-1)*width)-(((m+1)/(width*(m-1)))*total)
	'jval=round(jval,3)'	
	return jval





def sweep_n(data, minimum, maximum, min_bins, max_bins):
	list_out = []
	for i in range(min_bins, max_bins+1):
		hist,_,_ = plt.hist(data,i,(minimum,maximum))
		w= (maximum-minimum)/i		
		j=compute_j(hist,w)
		'j=round(j,3)'
		list_out.append(j)
	return (list_out)
	


def find_min(l):
    
	min_val = min(l)
	min_pos = l.index(min(l))
	opt_val = (min_val, min_pos)
	return opt_val
	"""
    generic function that takes a list of numbers and returns smallest number in that list its index.
    return optimal value and the index of the optimal value as a tuple.

    :param l: list
    :return: tuple
    """
 


if __name__ == '__main__':
    data = np.loadtxt('input.txt')  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
