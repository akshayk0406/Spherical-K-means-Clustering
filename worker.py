import os
files = ['bag.csv','char3.csv','char5.csv','char7.csv']
clusters = [20,40,60]


for j in [0,1]:
	if 0==j:
		cmd = "./preprocess"
	else:
		cmd = "./preprocess --use_tfidf=0"
	print cmd
	os.system(cmd)
	for i in [0,1]:
		for f in files:
			for clus in clusters:
				output_file = 'output/'
				output_file = output_file + f + '_' + str(clus) + '.txt'
				if 0 == i:
					cmd = './sphkmeans ' + f + ' newsgroups.class ' + str(clus) + ' 20 ' + output_file
					os.system(cmd)
				else:
					cmd = './sphkmeans ' + f + ' newsgroups.class ' + str(clus) + ' 20 ' + output_file + ' --method==inc'
					os.system(cmd)
				print cmd
		print "\n\n ================================= \n\n"
	print "\n\n **************************************** \n\n **************************************\n\n"
	
