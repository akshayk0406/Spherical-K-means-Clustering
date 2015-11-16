kmeans:
	gcc sphkmeans.c -lm -O2 -o sphkmeans
inc_kmeans:
	gcc inc_sphkmeans.c -lm -O2 -o inc_sphkmeans
preprocess:
	gcc preprocess.c -O2 -o preprocess
