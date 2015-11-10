import os
files = ['bag.csv','char3.csv','char5.csv','char7.csv']
clusters = [20,40,60]

for f in files:
	for clus in clusters:
		output_file = 'output/'
		output_file = output_file + f + '_' + str(clus) + '.txt'
		cmd = './sphkmeans ' + f + ' newsgroups.class ' + str(clus) + ' 20 ' + output_file
		print cmd
		os.system(cmd)
