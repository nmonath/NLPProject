import sys

def Organize(list_of_files):
	table_1 = ""
	table_2 = ""
	for f in list_of_files:
		for line in open(f, 'r'):
			if line.count('&') == 11:
				table_1 += line 
			if line.count('&') == 13:
				table_2 += line 
	print(table_1)

Organize(sys.argv[1:])

