#!/usr/bin/python3
year=2021
# Your code should be below this line
if((year%4) != 0):
	print 'false'
elif(year <= 0):
	print 'false'
else:
	if(year % 100 == 0):
		if(year%400 ==0):
			print 'true'
		else:
			print 'false'
	elif((year%4) == 0):
		print 'true' 

