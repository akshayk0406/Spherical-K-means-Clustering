#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<assert.h>

#define STOP_LIMIT 5
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
	int min = 0;
	int max = rowind-1;
	int idx = 0;	
	int ispresent = 0;

	int r;
    const unsigned int range = 1 + max - min;
    const unsigned int buckets = RAND_MAX / range;
    const unsigned int limit = buckets * range;

    do
    {
        r = rand();
		if(r<=limit)
		{
			r = min + (r/buckets);
			ispresent = 0;
			for(int i=0;i<idx && !ispresent;i++)
			{
				if(init_centroids[i]==r) ispresent = 1;
			}
			if(!ispresent) { init_centroids[idx] = r ; idx++;}
		}

    } while (idx<clusters);

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

int get_best_cluster(int req_object_id)
{
	int result = 0;
	double dist = 0.0;
	double max_similarity = -2.00;
	for(int i=0;i<clusters;i++)
	{
		dist = 0;
		for(int j=rowptr[req_object_id];j<rowptr[req_object_id+1];j++)
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
			objective_function[i][j]=0;
		}
		ans = ans + sqrt(dist);
	}
	return ans;	
}

int shouldBreak(int iter,double cur_obj_value)
{
	int is_solution_stable = 1;
	if( iter < STOP_LIMIT ) iter_objective_function[iter] = cur_obj_value;
	else
	{
		for(int i=0;i<STOP_LIMIT-1;i++) iter_objective_function[i] = iter_objective_function[i+1];
		iter_objective_function[STOP_LIMIT-1] = cur_obj_value;

		for(int i=0;i<STOP_LIMIT-1 && is_solution_stable ;i++)
		{
			if(mabs(iter_objective_function[i]-iter_objective_function[i+1]) > tolerance)
					is_solution_stable = 0;
		}
	}
	return iter >= STOP_LIMIT && is_solution_stable ? 1:0;
}

double do_clustering(int run)
{
	int cluster_id = 0;
	srand(seeds[run]);
	double pre_obj_value = 0.0;
	double cur_obj_value = 0.0;
	double csum = 0.0;	

	get_initial_centroids(seeds[run]);
	for(int i=0;i<clusters;i++)
	{
		object_id = init_centroids[i];
		csum = 0.0;
		for(int j=rowptr[object_id];j<rowptr[object_id+1];j++) 
		{
			centroids[i][colptr[j]] = values[j];
			csum = csum + values[j]*values[j];
		}
		normalizing_factor[i] = sqrt(csum);
	}

	for(int k=0;k<MAX_ITERATIONS;k++)
	{
		for(int i=0;i<rowind;i++)
		{
				cluster_id = get_best_cluster(i);
				cluster_map[i] = cluster_id;
				for(int j=rowptr[i];j<rowptr[i+1];j++) tcentroids[cluster_id][colptr[j]] = tcentroids[cluster_id][colptr[j]] + values[j];
				cluster_count[cluster_id] = cluster_count[cluster_id] + 1;
		}
		for(int i=0;i<clusters;i++)
		{
			csum = 0.0;
			for(int j=0;j<=TOTAL_FEATURES;j++) 
			{
				centroids[i][j] = tcentroids[i][j] / cluster_count[i];
				tcentroids[i][j]=0;
				csum = csum + centroids[i][j] * centroids[i][j];
			}
			normalizing_factor[i]  = sqrt(csum);
			cluster_count[i] = 0;
		}
		
		cur_obj_value = evaluate_objective_function();
		if(shouldBreak(k,cur_obj_value))
		{
				printf("Breaking at %d iteration\n",k+1);
				break;
		}
	}
	return cur_obj_value;
}

void write_output(char *fname)
{
	FILE* fp = fopen(fname,"w");
	for(int i=0;i<rowind;i++) fprintf(fp,"%d,%d\n",i,best_cluster_map[i]);
	fclose(fp);
}

void write_entropy_matrix()
{
	const char* entropy_fname = "entropy.txt";
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
        for(int j=0;j<total_classes;j++) rsum = rsum + entropy_matrix[i][j];
        for(int j=0;j<total_classes;j++) csum = max(csum , entropy_matrix[i][j]);
        ans = ans + (csum*rsum)/rowind;
    }
    return ans;
}

int main(int argc,char **argv)
{
	double max_obj_value = 0.0;
	double obj_value = 0.0;
	init();
	readInput(argv[1]);
	readClassFile(argv[2]);
	clusters = atoi(argv[3]); 
	trials = atoi(argv[4]);
	for(int i=0;i<trials;i++)
	{
		obj_value = do_clustering(i);
		if(obj_value > max_obj_value)
		{
			max_obj_value = obj_value;
			for(int j=0;j<rowind;j++) best_cluster_map[j] = cluster_map[j];
		}
	}

	write_output(argv[5]);
	for(int i=0;i<rowind;i++) 
		entropy_matrix[best_cluster_map[i]][class_map[i]] = entropy_matrix[best_cluster_map[i]][class_map[i]]+1;	

	write_entropy_matrix();	
	printf("############################\n");
	printf("Objective function value: %.6lf\n",max_obj_value);
	printf("Entropy: %.6lf\n",compute_entropy());
	printf("purity: %.6lf\n",compute_purity());
	printf("############################\n");
	return 0;
}
