#include <stdio.h>
#include <time.h>
#include "sha1.cu"
#include "tree.cu"

#define MESSAGE_SIZE 1
#define HASH_SIZE 20
#define MAX_ARITY 3

#define FILE_NAME "results.csv"


//parallel, GPU implementation of the merkle tree
//capable of generating merkle roots of the variable tree modes 
__global__ 
void hashTreeP 
(
	m_node   *nodes,
	uint64_t N,
	uint8_t  *arities,
	uint64_t *offsets,
	uint8_t  height,
	const unsigned char    *message
)
{
	unsigned char buffer[HASH_SIZE*MAX_ARITY];
	uint16_t thread = threadIdx.x;
	uint16_t block_size = blockDim.x;

	//calculate the message's hash
	for (uint64_t idx=thread; idx<N; idx+=block_size){
		for (uint8_t i=0;i<arities[1];i++)
			SHA1((buffer+(i*HASH_SIZE)),message,MESSAGE_SIZE);
		SHA1(nodes[idx].hash,buffer,HASH_SIZE*arities[1]);
	}
	__syncthreads();

	//parallel reduction begins, only one child will proceed
	//through each height level
	for (uint8_t i=2;i<=height;i++){
		for (uint64_t idx=thread;idx<N;idx+=block_size){
			if (idx%offsets[i]==0){
				for (uint8_t j=0;j<arities[i];j++){
					memcpy((buffer+(j*HASH_SIZE)),
							nodes[idx+j*offsets[i-1]].hash,
							HASH_SIZE);
				}
				SHA1(nodes[idx].hash,buffer,HASH_SIZE*arities[i]);
			}
		}
		__syncthreads();
	}
}

int checkArguments(int argc,char **argv)
{
	//extracting arguments
	if (argc != 5){
		printf("enter correct arguments\n");
		return -1;
	}
	if (atoi(argv[1]) <  10){
		printf("Enter a greater start block\n");
		return -1;
	}
	if (atoi(argv[2]) < atoi(argv[1])){
		printf("end blocks less than start blocks\n");
		return -1;
	}
	if (atoi(argv[3]) <= 0){
		printf("enter a greater granularity\n");
		return -1;
	}
	if (atoi(argv[4]) <= 0){
		printf("enter a greater number of tests\n");
		return -1;
	}
	return 1;
}


int main(int argc,char **argv)
{
	if (!checkArguments(argc,argv))
		return 0;

	const uint64_t startBlocks = atoi(argv[1]);
	const uint64_t endBlocks   = atoi(argv[2]);
	const uint64_t granularity = atoi(argv[3]);
	const uint64_t numTests    = atoi(argv[4]);

	//configure output file
	FILE *of;
	of = fopen(FILE_NAME,"w");
	fprintf(of,"Optimized vs Binary Merkle Tree Modes\n\n");
	fprintf(of,"Blocks, Binary Time, Optimized Time\n");
	fclose(of);

	//host and device trees;
	m_tree h_tree;
	m_tree d_tree;

	//collect timing metrics
	double start_t;
	double total_t;
	for (uint64_t i = startBlocks;i<endBlocks;i+=granularity){
		of = fopen(FILE_NAME,"a");
		fprintf(of,"%ld,  ",i);
		fclose(of);

		//Binary tree
		total_t = 0;
		for (uint64_t j=0;j<numTests;j++){
				//create host tree
			createBinaryTree(&h_tree,i,MESSAGE_SIZE);
				//copy host tree to device tree
			cudaCopyTree(&d_tree,&h_tree);
				//measure kernel execution time
			start_t = clock();
			hashTreeP<<<1, 1024>>>(
	 			d_tree.nodes,
	 			h_tree.endIdx[1] - h_tree.startIdx[1] + 1,
	 			d_tree.arities,
	 			d_tree.offsets,
	 			d_tree.height,
	 			d_tree.message
			);
			cudaDeviceSynchronize();
			total_t += (double)(clock() - start_t)/CLOCKS_PER_SEC;
				//free the trees
			cudaFreeTree(&d_tree);
			freeTree(&h_tree);
		}
		of = fopen(FILE_NAME,"a");
		fprintf(of,"%lf,  ",total_t/numTests);
		fclose(of);

		//Optimized tree
		total_t = 0;
		for (uint64_t j=0;j<numTests;j++){
				//create host tree
			createOptimizedTree(&h_tree,i,MESSAGE_SIZE);
				//copy host tree to device tree
			cudaCopyTree(&d_tree,&h_tree);
				//measure kernel execution time
			start_t = clock();
			hashTreeP<<<1, 1024>>>(
	 			d_tree.nodes,
	 			h_tree.endIdx[1] - h_tree.startIdx[1] + 1,
	 			d_tree.arities,
	 			d_tree.offsets,
	 			d_tree.height,
	 			d_tree.message
			);
			cudaDeviceSynchronize();
			total_t += (double)(clock() - start_t)/CLOCKS_PER_SEC;
				//free the trees
			cudaFreeTree(&d_tree);
			freeTree(&h_tree);
		}
		of = fopen(FILE_NAME,"a");
		fprintf(of,"%lf\n",total_t/numTests);
		fclose(of);
	}
	return 0;
}
