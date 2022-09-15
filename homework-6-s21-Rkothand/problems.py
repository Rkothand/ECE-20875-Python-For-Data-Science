import re

def problem1(searchstring):
	    
	p=re.compile(r"((\(\d{3}\)\s)|(\d{3}\-))?\d{3}\-\d{4}")
	
	if(p.fullmatch(searchstring)):
		return True
	else: 
		return False
	"""
    Match phone numbers.

    :param searchstring: string
    :return: True or False
    """

def problem2(searchstring):
	p=re.compile(r"(\d+\s)(([A-Z][a-z]*\s)+)(St\.|Ave\.|Dr\.|Rd\.)")
		
	
	return p.search(searchstring).group(2)[:-1]

	"""
    Extract street name from address.

    :param searchstring: string
    :return: string
    """
'''
(([A-Z][a-z])+)
(([A-Z][a-z]*\s)+)
p.search(searchstring).group(3)
'''
def problem3(searchstring):
	p=re.compile(r"(\d+\s)(([A-Z][a-z]*\s)+)(St\.|Ave\.|Dr\.|Rd\.)")
	
	b=p.sub
	m=p.search(searchstring).group(3)[::-1]
	m=m[1:]
	return p.sub(p.search(searchstring).group(1) + m + " "+p.search(searchstring).group(4),searchstring)
	"""
    Garble Street name.

    :param searchstring: string
    :return: string
    """
    


if __name__ == '__main__' :
    print(problem1('765-494-4600')) #True
    print(problem1(' 765-494-4600 ')) #False
    print(problem1('(765) 494 4600')) #False
    print(problem1('(765) 494-4600')) #True    
    print(problem1('494-4600')) #True
    
    print(problem2('The EE building is at 465 Northwestern Ave.')) #Northwestern
    print(problem2('Meet me at 201 South First St. at noon')) #South First
    
    print(problem3('The EE building is at 465 Northwestern Ave.'))
    print(problem3('Meet me at 201 South First St. at noon'))
