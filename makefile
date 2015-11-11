make kmeans:
	gcc sphkmeans.c -O2 -o sphkmeans
make inc_kmeans:
	gcc inc_sphkmeans.c -O2 -o inc_sphkmeans
make preprocess:
	gcc preprocess.cpp -O2 -o preprocess
