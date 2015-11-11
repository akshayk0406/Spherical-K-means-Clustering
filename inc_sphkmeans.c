#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<math.h>
#include<time.h>
#include<assert.h>

#define STOP_LIMIT 3
#define MAX_OBJECTS 10000
#define MAX_FEATURES 8000000
#define MAX_CENTROIDS 100
#define CLASS_NAME_LENGTH 64
#define MAX_SEEDS 20
#define MAX_ITERATIONS 100
#define square(x) x*x
#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define mabs(x) x<0?-x:x

char class_name[MAX_OBJECTS][CLASS_NAME_LENGTH];
int rowptr[MAX_OBJECTS];
int colptr[MAX_FEATURES];
int cluster_count[MAX_CENTROIDS];
int cluster_map[MAX_OBJECTS];
int pre_cluster_map[MAX_OBJECTS];
int best_cluster_map[MAX_OBJECTS];
int entropy_matrix[MAX_CENTROIDS][MAX_OBJECTS];
int class_map[MAX_OBJECTS];
int init_centroids[MAX_CENTROIDS];
double iter_objective_function[STOP_LIMIT];
double normalizing_factor[MAX_CENTROIDS];
double values[MAX_FEATURES];
double centroids[MAX_CENTROIDS][MAX_FEATURES];
double tcentroids[MAX_CENTROIDS][MAX_FEATURES];
double objective_function[MAX_CENTROIDS][MAX_FEATURES];

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
double frequency;
const double tolerance = 1e-5;

void init()
{
	int i = 0 ;
	for(i=1;i<=39;i=i+2) seeds[i/2] = i;
}

void printCSR()
{
	for(int i=0;i<=rowind;i++) 
	{
		for(int j=rowptr[i];j<rowptr[i+1];j++) printf("(%d %lf),",colptr[j],values[j]);
		printf("\n");
	}
}

void get_initial_centroids(int run)
{
	srand(run);
	int block_size = rowind/clusters;
	int n = 0;
	for(int i=0;i<clusters;i++)
	{
		n = rand()%block_size;
		init_centroids[i] = i*block_size+n;
	}
	return;
	n = rowind;
	int rem = RAND_MAX%n;
	int x = 0;
	int idx = 0;
	int ispresent = 0;
	while(idx<clusters)
	{
		do
		{
			x = rand();	
		}while(x >= RAND_MAX - rem);
		x = x%n;
		ispresent = 0;
		for(int i=0;i<idx && !ispresent;i++)
			if(init_centroids[i]==x) ispresent = 1;
		
		if(!ispresent) { init_centroids[idx] = x ; idx++;}
	}
}

void readInput(char *fname)
{
	pre_object_id = -1;
	FILE *fp = fopen(fname,"r");
	while(fscanf(fp,"%d,%d,%lf",&object_id,&feature_id,&frequency) !=EOF)
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

double evaluate_objective_function()
{
	for(int i=0;i<rowind;i++)
	{
		for(int j=rowptr[i];j<rowptr[i+1];j++) 
			objective_function[cluster_map[i]][j] = objective_function[cluster_map[i]][j] + values[j];	
	}
	double dist = 0.0,ans = 0.0;
	for(int i=0;i<clusters;i++)
	{
		dist = 0.0;
		for(int j=0;j<=TOTAL_FEATURES;j++) 
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
	double csum = 0.0;
	for(int j=0;j<=TOTAL_FEATURES;j++) csum = csum + centroids[cluster_id][j]*centroids[cluster_id][j];
	csum = sqrt(csum);
	assert(csum>=0);
	normalizing_factor[cluster_id] = csum;
}

int get_best_cluster(int req_object_id)
{
	double csum = 0.0;
	double best_sim = -2.00;
	int best_cluster_id = 0;
	for(int i=0;i<clusters;i++)
	{
		csum = 0.0;
		for(int j=rowptr[req_object_id];j<rowptr[req_object_id+1];j++)
			csum = csum + centroids[i][colptr[j]]*values[j];
		
		csum = csum/normalizing_factor[i];
		assert(csum>=0 && csum<=1.00);
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
	double p1 = 0;
	for(int j=rowptr[ptid];j<=rowptr[ptid+1];j++)
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

double do_clustering(int run)
{
	int cluster_id = 0;
	int pre_cluster_id = 0;
	srand(seeds[run]);
	double cur_obj_value = 0.0;
	double csum = 0.0;	
	double p1 = 0.0;

	get_initial_centroids(seeds[run]);
	for(int i=0;i<rowind;i++) cluster_map[i] = -1;
	for(int i=0;i<clusters;i++) for(int j=0;j<=TOTAL_FEATURES;j++) centroids[i][j] = 0;
	for(int i=0;i<clusters;i++)
	{
		cluster_id = init_centroids[i];
		cluster_map[cluster_id] = i;
		cluster_count[i] = 1;
		for(int j=rowptr[cluster_id];j<rowptr[cluster_id+1];j++) centroids[i][colptr[j]] = values[j];
		normalizing_factor[i] = 1.0;
	}
	
	int centroid_changed = 0;
	int shouldContinue = 0;	
	for(int k=0;k<MAX_ITERATIONS;k++)
	{
		centroid_changed = 0;
		shouldContinue = 0;
		for(int i=0;i<rowind;i++)
		{
			shouldContinue = 0;
			for(int j=0;j<clusters && !shouldContinue;j++) 
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
		if(!centroid_changed) 
		{
			printf("Breaking at %d iteration \n",k+1);
			break;
		}
	}
	return evaluate_objective_function();
}

void write_output(char *fname)
{
	FILE* fp = fopen(fname,"w");
	for(int i=0;i<rowind;i++) fprintf(fp,"%d,%d\n",i,best_cluster_map[i]);
	fclose(fp);
}

void write_entropy_matrix(char *entropy_fname)
{
	FILE *fp = fopen(entropy_fname,"w");
	for(int i=0;i<clusters;i++)
    {
        for(int j=0;j<total_classes;j++)
            fprintf(fp,"%d ",entropy_matrix[i][j]);
        fprintf(fp,"\n");
    }
	fclose(fp);
}

double compute_entropy()
{
	double ans = 0.0,rsum=0.0,csum=0.0;
	for(int i=0;i<clusters;i++)
	{
		rsum = 0.0;
		csum = 0.0;
		for(int j=0;j<total_classes;j++) rsum = rsum + entropy_matrix[i][j];
		for(int j=0;j<total_classes;j++) 
		{
			if(entropy_matrix[i][j]) csum = csum + (-1*(entropy_matrix[i][j]/rsum)*log2((entropy_matrix[i][j]/rsum)));
		}
		ans = ans + (csum*rsum)/rowind;
	}
	return ans;
}

double compute_purity()
{
	double ans = 0.0,rsum=0.0,csum=0.0;
    for(int i=0;i<clusters;i++)
    {
        rsum = 0.0;
        csum = 0.0;
        for(int j=0;j<total_classes;j++) csum = max(csum , entropy_matrix[i][j]);
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
	double obj_value = 0.0;
	double max_obj_value = 0.0;
	init();
	readInput(argv[1]);
	readClassFile(argv[2]);
	clusters = atoi(argv[3]); 
	trials = atoi(argv[4]);
	double t1,t2;
    struct timeval tv1,tv2;

	gettimeofday (&tv1, NULL);
    t1 = (double) (tv1.tv_sec) + 0.000001 * tv1.tv_usec;
	for(int i=0;i<trials;i++)
	{
		obj_value = do_clustering(i);
		if(obj_value > max_obj_value)
		{
			max_obj_value = obj_value;
			for(int j=0;j<rowind;j++) best_cluster_map[j] = cluster_map[j];
		}
	}
	gettimeofday (&tv2, NULL);
    t2 = (double) (tv2.tv_sec) + 0.000001 * tv2.tv_usec;

	write_output(argv[5]);
	for(int i=0;i<rowind;i++) 
		entropy_matrix[best_cluster_map[i]][class_map[i]] = entropy_matrix[best_cluster_map[i]][class_map[i]]+1;	

	char *entropy_fname = (char*)malloc(128*sizeof(char));
	get_entropy_file_name(entropy_fname,argv[1]);
	write_entropy_matrix(entropy_fname);

	char measures[] = "measures.csv";
	FILE* fp = fopen(measures,"a");
	double entropy = compute_entropy();	
	double purity = compute_purity();
	printf("############################\n");
	printf("Objective function value: %.6lf\n",max_obj_value);
	printf("Entropy: %.6lf\n",entropy);
	printf("purity: %.6lf\n",purity);
	printf("############################\n");
	fprintf(fp,"%s,%d,%d,%d,%d,%d,%.6lf,%.6lf,%.6lf\n",argv[1],clusters,trials,rowind,TOTAL_FEATURES,colind,entropy,purity,t2-t1);
	return 0;
}
