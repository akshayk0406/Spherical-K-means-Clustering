#include<stdlib.h>
#include<stdio.h>

int main()
{
	char cmd[] = "/usr/bin/python pre_process.py";
	printf("Pre-processing newsgroup data ....\n");
	system(cmd);
	return 0;
}
