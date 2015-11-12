#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<math.h>
#include<time.h>
#include<assert.h>

#define MAX_CLASSES 100
#define STOP_LIMIT 3
#define MAX_OBJECTS 6600
#define MAX_FEATURES 7000000
#define MAX_CENTROIDS 65
#define CLASS_NAME_LENGTH 64
#define MAX_SEEDS 20
#define MAX_ITERATIONS 50
#define square(x) x*x
#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define mabs(x) x<0?-x:x

char class_name[MAX_OBJECTS][CLASS_NAME_LENGTH];
int rowptr[MAX_OBJECTS];
int colptr[MAX_FEATURES];
int cluster_count[MAX_CENTROIDS];
int class_map[MAX_OBJECTS];
float values[MAX_FEATURES];

int *cluster_map;
int *pre_cluster_map;
int *best_cluster_map;
int *init_centroids;
float *normalizing_factor;
int **entropy_matrix;
float **objective_function;
float **centroids;

int TOTAL_FEATURES;
int rowind;
int colind;
int clusters;
int trials;
int seeds[MAX_SEEDS];
int total_classes;
int pre_object_id;
int object_id;
int feature_id;
float frequency;
const float tolerance = 1e-5;

void init()
{
	int i = 0 ;
	for(i=1;i<=39;i=i+2) seeds[i/2] = i;

	centroids = (float **)malloc((clusters+1)*sizeof(float*));
	for(i=0;i<clusters;i++) 
		centroids[i] = (float *)malloc((TOTAL_FEATURES+1)*sizeof(float));

	normalizing_factor = (float *)malloc(sizeof(float)*(clusters+1));

	objective_function = (float **)malloc(sizeof(float*)*(clusters+1));
	for(i=0;i<clusters;i++)
		objective_function[i] = (float *)malloc(sizeof(float)*(TOTAL_FEATURES+2));	

	entropy_matrix = (int **)malloc(sizeof(int*)*(clusters+1));
	for(i=0;i<clusters;i++)
		entropy_matrix[i] = (int*)malloc(sizeof(int)*(total_classes+1));

	init_centroids = (int *)malloc(sizeof(int)*(clusters+1));
	
	pre_cluster_map = (int *)malloc(sizeof(int)*(rowind+1));
	cluster_map = (int *)malloc(sizeof(int)*(rowind+1));
	best_cluster_map = (int *)malloc(sizeof(int)*(rowind+1));
}

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
}

void printCSR()
{
	int i=0,j=0;
	for(i=0;i<=rowind;i++) 
	{
		for(j=rowptr[i];j<rowptr[i+1];j++) printf("(%d %f),",colptr[j],values[j]);
		printf("\n");
	}
}

void get_initial_centroids(int run)
{
	srand(run);
	int block_size = rowind/clusters;
	int n = 0;
	int i=0;
	for(i=0;i<clusters;i++)
	{
		n = rand()%block_size;
		init_centroids[i] = i*block_size+n;
	}
}

void readInput(char *fname)
{
	pre_object_id = -1;
	FILE *fp = fopen(fname,"r");
	while(fscanf(fp,"%d,%d,%f",&object_id,&feature_id,&frequency) !=EOF)
	{
		if(pre_object_id != object_id)
		{
			rowptr[rowind] = colind;
			rowind = rowind +1;
		}
		TOTAL_FEATURES = max(TOTAL_FEATURES,feature_id);
		colptr[colind] = feature_id;
		values[colind] = frequency;
		assert(frequency<=1.00);
		colind++;
		pre_object_id = object_id;
	}
	rowptr[rowind] = colind;
	fclose(fp);
}

void readClassFile(char *fname)
{
	FILE* fp = fopen(fname,"r");
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
		for(j=0;j<=TOTAL_FEATURES;j++) 
		{
			dist = dist + (objective_function[i][j]*objective_function[i][j]);
			assert(dist>=0);
			objective_function[i][j]=0;
		}
		ans = ans + sqrt(dist);
	}
	return ans;	
}

void normalize(int cluster_id)
{
	int j=0;
	float csum = 0.0;
	for(j=0;j<=TOTAL_FEATURES;j++) csum = csum + centroids[cluster_id][j]*centroids[cluster_id][j];
	csum = sqrt(csum);
	assert(csum>=0);
	normalizing_factor[cluster_id] = csum;
}

int get_best_cluster(int req_object_id)
{
	int i=0,j=0;
	float csum = 0.0;
	float best_sim = -2.00;
	int best_cluster_id = 0;
	for(i=0;i<clusters;i++)
	{
		csum = 0.0;
		for(j=rowptr[req_object_id];j<rowptr[req_object_id+1];j++)
			csum = csum + centroids[i][colptr[j]]*values[j];
		
		csum = csum/normalizing_factor[i];
		assert(csum+tolerance>=0.00 && csum-tolerance<=1.00);
		if(csum + tolerance > best_sim)
		{
			best_sim  = csum;
			best_cluster_id = i;
		}
	}
	return best_cluster_id;
}


void MovePoint(int cluster_id,int ptid,int fg)
{
	int j=0;
	float p1 = 0;
	for(j=rowptr[ptid];j<=rowptr[ptid+1];j++)
	{
		p1 = centroids[cluster_id][colptr[j]]*cluster_count[cluster_id];
		if(!fg) 
		{
			p1 = p1 - values[j];
			cluster_count[cluster_id] = cluster_count[cluster_id] - 1;
		}
		else 
		{
			p1 = p1 + values[j];
			cluster_count[cluster_id] = cluster_count[cluster_id] + 1;
		}
		assert(cluster_count[cluster_id]>0);
		centroids[cluster_id][colptr[j]] = p1/cluster_count[cluster_id];
	}	
	normalize(cluster_id);
}

float do_clustering(int run)
{
	int cluster_id = 0;
	int pre_cluster_id = 0;
	srand(seeds[run]);
	float cur_obj_value = 0.0;
	float csum = 0.0;	
	float p1 = 0.0;
	int i=0,j=0,k=0;
	
	get_initial_centroids(seeds[run]);
	for(i=0;i<rowind;i++) cluster_map[i] = -1;
	for(i=0;i<clusters;i++) for(j=0;j<=TOTAL_FEATURES;j++) centroids[i][j] = 0;
	for(i=0;i<clusters;i++)
	{
		cluster_id = init_centroids[i];
		cluster_map[cluster_id] = i;
		cluster_count[i] = 1;
		for(j=rowptr[cluster_id];j<rowptr[cluster_id+1];j++) centroids[i][colptr[j]] = values[j];
		normalizing_factor[i] = 1.0;
	}
	
	int centroid_changed = 0;
	int shouldContinue = 0;	
	for(k=0;k<MAX_ITERATIONS;k++)
	{
		centroid_changed = 0;
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
				centroid_changed++;
				MovePoint(cluster_id,i,1);
			}
			else
			{
				if(pre_cluster_map[i]!=cluster_map[i])
				{
					centroid_changed++;
					MovePoint(pre_cluster_map[i],i,0);
					MovePoint(cluster_id,i,1);
				}
			}
			pre_cluster_map[i] = cluster_map[i];
		}
		if(!centroid_changed) break;
	}
	return evaluate_objective_function();
}

void write_output(char *fname)
{
	int i=0;
	FILE* fp = fopen(fname,"w");
	for(i=0;i<rowind;i++) fprintf(fp,"%d,%d\n",i,best_cluster_map[i]);
	fclose(fp);
}

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
		//csum = csum/log2(total_classes);
		ans = ans + (csum*rsum)/rowind;
	}
	return ans;
}

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

int main(int argc,char **argv)
{
	float obj_value = 0.0;
	float max_obj_value = 0.0;
	float t1,t2;
    struct timeval tv1,tv2;
	int i=0,j=0;	

	readInput(argv[1]);
	readClassFile(argv[2]);
	clusters = atoi(argv[3]); 
	trials = atoi(argv[4]);
	init();

	gettimeofday (&tv1, NULL);
    t1 = (float) (tv1.tv_sec) + 0.000001 * tv1.tv_usec;
	
	for(i=0;i<trials;i++)
	{
		obj_value = do_clustering(i);
		if(obj_value > max_obj_value)
		{
			max_obj_value = obj_value;
			for(j=0;j<rowind;j++) best_cluster_map[j] = cluster_map[j];
		}
	}
	
	gettimeofday (&tv2, NULL);
    t2 = (float) (tv2.tv_sec) + 0.000001 * tv2.tv_usec;

	write_output(argv[5]);
	for(i=0;i<rowind;i++) 
		entropy_matrix[best_cluster_map[i]][class_map[i]] = entropy_matrix[best_cluster_map[i]][class_map[i]]+1;	

	char *entropy_fname = (char*)malloc(128*sizeof(char));
	get_entropy_file_name(entropy_fname,argv[1]);
	write_entropy_matrix(entropy_fname);

	char measures[] = "measures.csv";
	FILE* fp = fopen(measures,"a");
	float entropy = compute_entropy();	
	float purity = compute_purity();
	printf("####################################################################  \n");
    printf("Objective function value: %.6f ,Entropy: %.6f ,Purity: %.6f ,#Rows: %d ,#Columns: %d ,#NonZeros: %d\n",max_obj_value,entropy,purity,rowind,TOTAL_FEATURES,colind);
    printf("####################################################################  \n");
	fprintf(fp,"%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",argv[1],clusters,trials,rowind,TOTAL_FEATURES,colind,entropy,purity,t2-t1);
	deinit();
	return 0;
}
