make kmeans:
	gcc sphkmeans.c -lm -O2 -o sphkmeans
make inc_kmeans:
	gcc inc_sphkmeans.c -lm -O2 -o inc_sphkmeans
make preprocess:
	gcc preprocess.cpp -O2 -o preprocess
