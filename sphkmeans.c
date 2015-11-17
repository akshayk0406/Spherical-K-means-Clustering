/************** Author: Akshay Kulkarni *****************/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<math.h>
#include<time.h>
#include<assert.h>

#define STOP_LIMIT 3
#define MAX_OBJECTS 7000
#define MAX_NON_ZEROS 7000000 
#define MAX_CENTROIDS 100
#define CLASS_NAME_LENGTH 64
#define MAX_SEEDS 20
#define MAX_ITERATIONS 100
#define square(x) x*x
#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define mabs(x) x<0?-x:x

/*
rowptr,colptr,values are the matrices for storing document-term in sparse format.
http://www.netlib.org/utk/people/JackDongarra/etemplates/node373.html
*/

int *rowptr;
int *colptr;
float *values;

/*
Arrays for storing mapping of object_id and class name
*/
char class_name[MAX_OBJECTS][CLASS_NAME_LENGTH];
int class_map[MAX_OBJECTS];

/*
cluster_count -> stores number of elements belonging to cluster i
cluster_map -> stores the index of the cluster to which ith object belong
pre_cluster_map -> stores the cluster mapping of previous occasion
best_cluster_map -> stores best cluster mapping (best determined by maximising objective function)
init_centroids -> stores the object_id's which are initialized are initial centroids
normalizing_factor -> store the normalizling factor for ith cluster
entropy_matrix -> stores the entropy matrix (#clusters*#classes). entropy_matrix[i][j] denotes number of items of class j belonging cluster i
objective_function -> stores vector addition of documents belonging to same cluster i.e objective_function[i] is a vector which sum of all vectors belonging to ith cluster
centroids -> stores current centroids
*/
int *cluster_count;
int *cluster_map;
int *pre_cluster_map;
int *best_cluster_map;
int *init_centroids;
float *normalizing_factor;
float *normalizing_points;
int **entropy_matrix;
float **objective_function;
float **centroids;

/*
Auxillary variables:-
total_features -> features space i.e number of attributes
rowind -> total number of documents
clusters -> number of clusters to find
seeds -> stores seed for generating inital centroids
*/
int total_features;
int rowind;
int colind;
int clusters;
int trials;
int seeds[MAX_SEEDS];
int total_classes;
int pre_object_id;
int object_id;
int feature_id;
int method;
float frequency;
const float tolerance = 1e-5;

/*
:purpose
	- to do all dynamic allocation of memory
*/
void init()
{
	int i = 0 ;
	for(i=1;i<=39;i=i+2) seeds[i/2] = i;
	
	centroids = (float **)calloc(clusters,sizeof(float*));
    for(i=0;i<clusters;i++)
        centroids[i] = (float *)calloc(total_features+1,sizeof(float));

	objective_function = (float **)calloc(clusters+1,sizeof(float*));
    for(i=0;i<clusters;i++)
        objective_function[i] = (float *)calloc(total_features+2,sizeof(float));

    entropy_matrix = (int **)calloc(clusters+1,sizeof(int*));
    for(i=0;i<clusters;i++)
        entropy_matrix[i] = (int*)calloc(total_classes+1,sizeof(int));

    normalizing_factor = (float *)calloc(clusters+1,sizeof(float));
    normalizing_points = (float *)calloc(clusters+1,sizeof(float));
    init_centroids = (int *)calloc(clusters+1,sizeof(int));
    pre_cluster_map = (int *)calloc(rowind+1,sizeof(int));
    cluster_map = (int *)calloc(rowind+1,sizeof(int));
    best_cluster_map = (int *)calloc(rowind+1,sizeof(int));
	cluster_count = (int*)calloc(clusters+1,sizeof(int));
}

/*
:purpose
	- to free all the dynamically allocated memory
*/
void deinit()
{
	int i=0;
	for(i=0;i<clusters;i++)
	{
		free(centroids[i]);
		free(objective_function[i]);
		free(entropy_matrix[i]);
	}

	free(centroids);
	free(objective_function);
	free(entropy_matrix);

	free(init_centroids);
	free(pre_cluster_map);
	free(cluster_map);
	free(best_cluster_map);
	free(normalizing_factor);
	free(rowptr);
	free(colptr);
	free(values);
	free(normalizing_points);
}

/*
:purpose
	- to print sparse matrix stored in CSR format.
*/
void printCSR()
{
	int i=0,j=0;
	for(i=0;i<=rowind;i++) 
	{
		for(j=rowptr[i];j<rowptr[i+1];j++) printf("(%d %.4lf),",colptr[j],values[j]);
		printf("\n");
	}
}

/*
:purpose
	- normalize points
*/

void normalize_points()
{
	int i=0,j=0;
	double csum = 0.0;
	for(i=0;i<rowind;i++)
	{
		csum = 0.0;
		for(j=rowptr[i];j<rowptr[i+1];j++) csum = csum + values[j]*values[j];
		normalizing_points[i] = sqrt(csum);
		for(j=rowptr[i];j<rowptr[i+1];j++) values[j] /= normalizing_points[i];
	}
}

/*
:purpose
	-  to generate initial centroids for k-means clustering
:param
	- run -> iteration number. seed is choosen accordingly
:result
	- stores the object_ids that form centroid in init_centroids
*/
void get_initial_centroids(int run)
{
	srand(run);
	int n = rowind;
	int rem = RAND_MAX%n;
	int x = 0;
	int idx = 0;
	int ispresent = 0;
	int i=0;
	while(idx<clusters)
	{
		do
		{
			x = rand();	
		}while(x >= RAND_MAX - rem);
		x = x%n;
		ispresent = 0;
		for(i=0;i<idx && !ispresent;i++)
			if(init_centroids[i]==x) ispresent = 1;		
		if(!ispresent) { init_centroids[idx] = x ; idx++;}
	}
}
/*
:purpose
	-  to read input data in the format of (object_id,feature_id,frequency)
:param
	- fname -> file name to read from
:result
	- populates rowptr,colptr and values with required values
*/
void allocateAndRead(char *fname)
{
	colptr = (int*)calloc(colind+1,sizeof(int));
	values = (float*)calloc(colind+1,sizeof(float));
	rowptr = (int*)calloc(rowind+1,sizeof(int));

	rowind = 0,colind = 0;	
	pre_object_id = -1;
	FILE *fp = fopen(fname,"r");
	if(!fp) return;
	while(fscanf(fp,"%d,%d,%f",&object_id,&feature_id,&frequency) !=EOF)
	{
		if(pre_object_id != object_id)
		{
			rowptr[rowind] = colind;
			rowind = rowind +1;
		}
		total_features = max(total_features,feature_id);
		colptr[colind] = feature_id;
		values[colind] = frequency;
		colind++;
		pre_object_id = object_id;
	}
	rowptr[rowind] = colind;
	fclose(fp);
}

/*
:purpose
	-  to read input data in the format of (object_id,feature_id,frequency) and allocate appropriate memory
:param
	- fname -> file name to read from
:result
	- populates rowptr,colptr and values with required values
*/
void readInput(char *fname)
{
	pre_object_id = -1;
	FILE *fp = fopen(fname,"r");
	if(!fp) return;
	while(fscanf(fp,"%d,%d,%f",&object_id,&feature_id,&frequency) !=EOF)
	{
		if(pre_object_id != object_id) 
			rowind++;
		total_features = max(total_features,feature_id);
		colind++;
		pre_object_id = object_id;
	}
	fclose(fp);
	allocateAndRead(fname);
}

/*
:purpose
	- to read class file containing mapping between object_id and class name
:param
	- fname -> file name to read from
:result
	- populates class_map and class_name with object_id and class name
*/
void readClassFile(char *fname)
{
	FILE* fp = fopen(fname,"r");
	if(!fp) return;
	int i = 0;
	int diff = 1;
	while(fscanf(fp,"%d,%s",&class_map[i],class_name[i])!=EOF) 
	{
		if(i>0)
		{
			if(strcmp(class_name[i],class_name[i-1]))
			{
				class_map[i] = diff;
				diff++;
			}
			else class_map[i] = class_map[i-1];
		}
		i++;
	}
	total_classes = diff;
	fclose(fp);
}

/*
:purpose
	- to find best cluster for given object
:param
	- req_object_id -> objected id for which we need to find best centroid
:result
	- return index of best cluster i.e the cluster with maximum cosine similarity
*/
int get_best_cluster(int req_object_id)
{
	int result = 0,i=0,j=0;
	float dist = 0.0;
	float max_similarity = -2.00;
	for(i=0;i<clusters;i++)
	{
		dist = 0.0;
		for(j=rowptr[req_object_id];j<rowptr[req_object_id+1];j++)
			dist = dist + centroids[i][colptr[j]]  * values[j] ;
	
		dist = dist/normalizing_factor[i];
		if( dist - max_similarity > tolerance )
		{
			max_similarity = dist;
			result = i;
		}
	}
	return result;
}

/*
:purpose
	- to evaluate objective function
:result
	- computes and return value of optimization function
*/
float evaluate_objective_function()
{
	int i=0,j=0;
	for(i=0;i<rowind;i++)
	{
		for(j=rowptr[i];j<rowptr[i+1];j++) 
			objective_function[cluster_map[i]][colptr[j]] = objective_function[cluster_map[i]][colptr[j]] + values[j];	
	}
	float dist = 0.0,ans = 0.0;
	for(i=0;i<clusters;i++)
	{
		dist = 0.0;
		for(j=0;j<=total_features;j++) 
		{
			dist = dist + (objective_function[i][j]*objective_function[i][j]);
			objective_function[i][j]=0;
		}
		ans = ans + sqrt(dist);
	}
	return ans;	
}

/*
:purpose
	- to normalize the cluster/centroid
:param
	- cluster_id
:result
	- normalises the cluster with given cluster_id
*/
void normalize_traditional(int cluster_id)
{
	int j=0;
	float csum = 0.0;
	for(j=0;j<=total_features;j++) 
	{
		centroids[cluster_id][j] /= cluster_count[cluster_id];
		csum = csum + centroids[cluster_id][j]*centroids[cluster_id][j];
	}
	csum = sqrt(csum);
	normalizing_factor[cluster_id] = csum;
}

/*
:purpose
	- to normalize the cluster/centroid
:param
	- cluster_id
:result
	- normalises the cluster with given cluster_id
*/
void normalize_incremental(int cluster_id)
{
	int j=0;
	float csum = 0.0;
	for(j=0;j<=total_features;j++) csum = csum + centroids[cluster_id][j]*centroids[cluster_id][j];
	csum = sqrt(csum);
	normalizing_factor[cluster_id] = csum;
}

/*
:pupose
	- compute entropy of the solution
:result:
	- return entropy of the solution
*/
float compute_entropy()
{
	int i=0,j=0;
	float ans = 0.0,rsum=0.0,csum=0.0;
	for(i=0;i<clusters;i++)
	{
		rsum = 0.0;
		csum = 0.0;
		for(j=0;j<total_classes;j++) rsum = rsum + entropy_matrix[i][j];
		for(j=0;j<total_classes;j++) 
		{
			if(entropy_matrix[i][j]) csum = csum + (-1*(entropy_matrix[i][j]/rsum)*log2((entropy_matrix[i][j]/rsum)));
		}
		//csum = csum / log2(total_classes);
		ans = ans + (csum*rsum)/rowind;
	}
	return ans;
}

/*
:pupose
	- compute purity of the solution
:result:
	- return purity of the solution
*/
float compute_purity()
{
	int i=0,j=0;
	float ans = 0.0,rsum=0.0,csum=0.0;
    for(i=0;i<clusters;i++)
    {
        rsum = 0.0;
        csum = 0.0;
        for(j=0;j<total_classes;j++) csum = max(csum , entropy_matrix[i][j]);
        ans = ans + csum/rowind;
    }
    return ans;
}

/*
:purpose
    - Initialzing  parameters for clustering
:param
	- run
:result
	- Initializes variables for next clustering sequence
*/

void initClusteringParameters(int run)
{
    int i=0,j=0;
    get_initial_centroids(seeds[run]); //populates initial centroids
	/* initializing the variables fro computation */
    for(i=0;i<rowind;i++)
    {
        cluster_map[i] = -1;
        pre_cluster_map[i] = -1;
    }

    for(i=0;i<clusters;i++)
    {
        cluster_count[i] = 0;
        for(j=0;j<=total_features;j++) centroids[i][j] = 0;
    }
	
	/* setting initial centroids */
	for(i=0;i<clusters;i++)
	{
		object_id = init_centroids[i];
		for(j=rowptr[object_id];j<rowptr[object_id+1];j++) 
			centroids[i][colptr[j]] = values[j];
		normalize_incremental(i);
		if(1==method) 
		{
			cluster_count[i] = 1;
			cluster_map[object_id] = i;
		}
	}
}

void MovePoint(int cluster_id,int ptid,int fg)
{
	int j=0;
	float p1 = 0;
	int req_cc = cluster_count[cluster_id];

	if(!fg) req_cc--;
	else req_cc++;

	for(j=rowptr[ptid];j<=rowptr[ptid+1];j++)
	{
		p1 = centroids[cluster_id][colptr[j]]*cluster_count[cluster_id];
		if(!fg) p1 = p1 - values[j];
		else p1 = p1 + values[j];
		centroids[cluster_id][colptr[j]] = p1/req_cc;
	}
	cluster_count[cluster_id] = req_cc;
	normalize_incremental(cluster_id);
}

int clusterAssignment(int iter)
{
	int i=0,changes=0,cluster_id=0;
	for(i=0;i<rowind;i++)
    {
		cluster_id = get_best_cluster(i); //get best cluster for ith object
        cluster_map[i] = cluster_id; //update cluster_map to store cluster id for ith object
        cluster_count[cluster_id] = cluster_count[cluster_id] + 1;
        if(pre_cluster_map[i]!=cluster_map[i] || 0==iter) changes++;
        pre_cluster_map[i] = cluster_map[i];
    }
	return changes;
}

void ComputeCentroids()
{
	int i=0,j=0;
	for(i=0;i<clusters;i++) for(j=0;j<=total_features;j++) centroids[i][j] = 0;
    for(i=0;i<rowind;i++)
   	{
		for(j=rowptr[i];j<rowptr[i+1];j++)
			centroids[cluster_map[i]][colptr[j]] = centroids[cluster_map[i]][colptr[j]]+values[j];
    }
        //normalizing centroids and copying tcentroids to centroids so that we have correct values for next iteration
  	for(i=0;i<clusters;i++)
    {
		normalize_traditional(i);
        cluster_count[i] = 0;
   	}
}

/*
:pupose
	- to do the clustering using incremental kmeans
:param
	- run -> iteration id
:result
	- value of objective function
*/

float incremental_kmeans(int run)
{
	int cluster_id = 0;
	int pre_cluster_id = 0;
	srand(seeds[run]);
	float cur_obj_value = 0.0;
	float csum = 0.0;
	float p1 = 0.0;
	int i=0,j=0,k=0;
	const int threshold = (0.1*(double)rowind)/100;

	initClusteringParameters(run);

	int changes = 0;
	int shouldContinue = 0;

	for(k=0;k<MAX_ITERATIONS;k++)
	{
		changes = 0;
		shouldContinue = 0;
		for(i=0;i<rowind;i++)
		{	
			shouldContinue = 0;
			for(j=0;j<clusters && !shouldContinue;j++)
				if(init_centroids[j] == i ) shouldContinue = 1;
			if(shouldContinue) continue;

			cluster_id = get_best_cluster(i);
			cluster_map[i] = cluster_id;
			if(0==k)
			{
				changes++;
				MovePoint(cluster_id,i,1);
			}
			else
			{
				if(pre_cluster_map[i]!=cluster_map[i])
				{
					changes++;
					MovePoint(pre_cluster_map[i],i,0);
					MovePoint(cluster_id,i,1);
				}
			}
			pre_cluster_map[i] = cluster_map[i];
		}
		if(changes<=threshold) break;
	}
	return evaluate_objective_function();
}

/*
:pupose
	- to do the clustering using traditional kmeans
:param
	- run -> iteration id
:result
	- value of objective function
*/
float traditional_kmeans(int run)
{
	int cluster_id = 0,i=0,j=0,k=0;
	srand(seeds[run]);
	float cur_obj_value = 0.0;
	float csum = 0.0;	
	float p1 = 0.0;
	const int threshold = (0.3*(double)rowind)/100;
	int changes = 0;
	
	initClusteringParameters(run);
	
	for(k=0;k<MAX_ITERATIONS;k++)
	{
		changes = clusterAssignment(k);
		ComputeCentroids();
		//Breaking when number of points changing centroids is less than certain threshold
		if(changes<=threshold) break; 
	}
	return evaluate_objective_function();
}

/*
:purpose
	- to write clustering solution to file
:param
	- fname -> writes clustering solution to file fname
*/
void write_output(char *fname)
{
	int i=0;
	FILE* fp = fopen(fname,"w");
	if(!fp) return;
	for(i=0;i<rowind;i++) fprintf(fp,"%d,%d\n",i,best_cluster_map[i]);
	fclose(fp);
}

/*
:pupose
	- to write entropy matrix (#clusters * #classes)
:param
	- entropy_fname -> file name where to write the solution
*/
void write_entropy_matrix(char *entropy_fname)
{
	int i=0,j=0;
	FILE *fp = fopen(entropy_fname,"w");
	for(i=0;i<clusters;i++)
    {
        for(j=0;j<total_classes;j++)
            fprintf(fp,"%d ",entropy_matrix[i][j]);
        fprintf(fp,"\n");
    }
	fclose(fp);
}

char *get_entropy_file_name(char* entropy_fname,char *fname)
{
	char clus[10],trial[10];
	sprintf(clus,"%d",clusters);
	sprintf(trial,"%d",trials);
	strcat(entropy_fname,"entropy_matrix/");
	strcat(entropy_fname,fname);
	strcat(entropy_fname,"_clusters_");
	strcat(entropy_fname,clus);
	strcat(entropy_fname,"_trials_");
	strcat(entropy_fname,trial);
	strcat(entropy_fname,".txt");
	return entropy_fname;
}

/*
:purpose
	- setting clustering parameters
:param
	- command line arguments
:result
	- setting method variable to run incremental vs traditional kmeans
*/
void setParameters(char *params)
{
	method = 0;
	if(strstr(params,"method") && strstr(params,"inc")) //doing incremental clustering
		method = 1;
}

int main(int argc,char **argv)
{
	if(argc>=6)
	{
		int i=0,j=0;
		float obj_value = 0.0;
		float max_obj_value = 0.0;
		double t1,t2;
    	struct timeval tv1,tv2;

		readInput(argv[1]); //reading input file
		readClassFile(argv[2]); //reading class file
		clusters = atoi(argv[3]);  // number of clusters
		trials = atoi(argv[4]); //number of trials
		
		init();
		if(argc>=7) setParameters(argv[6]);
		normalize_points();	
		gettimeofday (&tv1, NULL);
    	t1 = (double) (tv1.tv_sec) + 0.000001 * tv1.tv_usec;
	
		for(i=0;i<trials;i++)
		{
			if(!method) obj_value = traditional_kmeans(i);
			else obj_value = incremental_kmeans(i);
			if(obj_value>max_obj_value)
			{
				max_obj_value = obj_value;
				//storing best clustering mapping
				for(j=0;j<rowind;j++) 
					best_cluster_map[j] = cluster_map[j];
			}
		}
		gettimeofday (&tv2, NULL);
    	t2 = (double) (tv2.tv_sec) + 0.000001 * tv2.tv_usec;

		write_output(argv[5]);
		for(i=0;i<rowind;i++) 
			entropy_matrix[best_cluster_map[i]][class_map[i]] = entropy_matrix[best_cluster_map[i]][class_map[i]]+1;	
		
		char *entropy_fname = (char*)malloc(128*sizeof(char));
		strcpy(entropy_fname,strcat(argv[5],".entropy"));
		write_entropy_matrix(entropy_fname);

		float entropy = compute_entropy();	
		float purity = compute_purity();
		//Printing entropy matrix
		for(i=0;i<clusters;i++)
		{
			for(j=0;j<total_classes;j++)
				printf("%d ",entropy_matrix[i][j]);
			printf("\n");
		}
	
		printf("####################################################################  \n\n");
		printf("Objective function value: %.6f ,Entropy: %.6f ,Purity: %.6f ,#Rows: %d ,#Columns: %d ,#NonZeros: %d, time(in secs): %.6lf\n",max_obj_value,entropy,purity,rowind,total_features,colind,t2-t1);
		printf("\n####################################################################  \n\n");
	}
	else
	{
		printf("Wrong Input Format. Input format should be like 'input-file class-file #clusters #trials output-file'\n");
	}
	return 0;
}
