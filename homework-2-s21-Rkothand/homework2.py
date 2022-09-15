def histogram(data, n, b, h):
    # data is a list
    # n is an integer
    # b and h are floats
    
    # Write your code here
	hist = []
	if n <=0 or h<b:
		return hist
	hist = [0]*n #Initialize the histogram `hist` as a list of `n` zeros.
	w=(h-b)/n
	for i in data:
		if( i > b and i<h):
			for j in range(n): 		
				if i >=(b+(j*w)) and i<(b+(j+1)*w):
					hist[j]+=1
	return hist
			
    # return the variable storing the histogram
    # Output should be a list
#
   #pass


def addressbook(name_to_phone, name_to_address):
    #name_to_phone and name_to_address are both dictionaries
    
    # Write your code here
	address_to_all = {}
	for k,v in name_to_address.items():
		if(v not in address_to_all):
			address_to_all[v]=([k],name_to_phone[k])		
		else:
			address_to_all[v][0].append(k)
			if(name_to_phone[k] != address_to_all[v][1]):
				print('Warning: '+ k + ' has a different number for '+v+' than '+address_to_all[v][0][0]+ '. Using the number for ' + address_to_all[v][0][0])
	return address_to_all
    # return the variable storing address_to_all
    # Output should be a dictionary
    
  # pass
#
