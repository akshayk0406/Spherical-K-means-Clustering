#include<stdlib.h>
#include<stdio.h>
#include<string.h>

int main(int argc,char **argv)
{
	char *cmd = (char*)malloc(128*sizeof(char));
	strcpy(cmd,"/usr/bin/python pre_process.py ");
	if(argc>=2)
		strcat(cmd,argv[1]);
	printf("Running command %s\n",cmd);
	system(cmd);
	free(cmd);
	return 0;
}
