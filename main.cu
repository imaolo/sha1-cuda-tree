#include <stdio.h>
#include <time.h>
#include "sha1.cu"
#include "tree.cu"

#define MESSAGE_SIZE 1
#define HASH_SIZE 20
#define MAX_ARITY 3

#define FILE_NAME "results.csv"


//prints HASH_SIZE charcters of the string in HEX form
void printHash(const unsigned char *hash)
{
	for (int i=0;i<HASH_SIZE;i++)
		printf("%02x",hash[i]);
	printf("\n");
}

//compares two hash strings. 0 for unequal, 1 for equal
uint8_t cmpHash
(
	const unsigned char *a_hash,
	const unsigned char *b_hash
)
{
	for(int i=0;i<HASH_SIZE;i++){
		if (a_hash[i] != b_hash[i])
			return 0;
	}
	return 1;
}


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


//returns the time taken to compute the merkle root of the tree
//returns -1 if the trees are not equal
__host__ double testTree(m_tree *h_tree,m_tree *d_tree)
{
	//timing variables;
	clock_t start_t;
	double total_t;
	//invoke kernel
	start_t = clock();
	hashTreeP<<<1, 1024>>>(
	 	d_tree->nodes,
	 	h_tree->endIdx[1] - h_tree->startIdx[1] + 1,
	 	d_tree->arities,
	 	d_tree->offsets,
	 	d_tree->height,
	 	d_tree->message
	);
	cudaDeviceSynchronize();
	total_t = (double)(clock() - start_t)/CLOCKS_PER_SEC;
	//error checking
	unsigned char d_merkle_root[HASH_SIZE];
	cudaMemcpy(
		d_merkle_root,
		d_tree->nodes[0].hash,
		HASH_SIZE*sizeof(unsigned char),
		cudaMemcpyDeviceToHost
	);
	hashTreeS(h_tree);
	if (!cmpHash(d_merkle_root,h_tree->nodes[0].hash))
		return -1.0f;

	return total_t;
}

int main(int argc,char **argv)
{
	//extracting arguments
	if (argc != 2){
		printf("enter correct arguments\n");
		return 0;
	}
	if (atoi(argv[1]) <  10){
		printf("Enter a greater max blocks\n");
		return 0;
	}
	const uint64_t maxBlocks = atoi(argv[1]);

	//configure output file
	FILE *of;
	of = fopen(FILE_NAME,"w");
	fprintf(of,"Optimized vs Binary Merkle Tree Modes\n\n");
	fprintf(of,"Blocks, Binary Time, Optimized Time\n");
	fclose(of);

	//host and device trees;
	m_tree h_tree;
	m_tree d_tree;
	//timing varible
	double runtime;

	//collect timing metrics
	for (uint64_t numBlocks = 10;numBlocks<maxBlocks;numBlocks+=25){
		of = fopen(FILE_NAME,"w");
		fprintf(of,"%ld,",numBlocks);
		fclose(of);
		//Binary tree
			//create host tree
		createBinaryTree(&h_tree,numBlocks,MESSAGE_SIZE);
			//copy host tree to device tree
		cudaCopyTree(&d_tree,&h_tree);
			//test the tree for speed and correctness
		if ((runtime = testTree(&h_tree,&d_tree)) < 0){
			printf("FAILED\n");
			cudaFreeTree(&d_tree);
			freeTree(&h_tree);
			return 0;
		}
		else{
			of = fopen(FILE_NAME,"w");
			fprintf(of,"%lf,",runtime);
			fclose(of);
		}
			//free the trees
		cudaFreeTree(&d_tree);
		freeTree(&h_tree);

		//Optimized tree
			//create host tree
		createOptimizedTree(&h_tree,numBlocks,MESSAGE_SIZE);
			//copy host tree to device tree
		cudaCopyTree(&d_tree,&h_tree);
			//test the tree for speed and correctness
		if ((runtime = testTree(&h_tree,&d_tree)) < 0){
			printf("FAILED\n");
			cudaFreeTree(&d_tree);
			freeTree(&h_tree);
			return 0;
		}
		else{
			of = fopen(FILE_NAME,"w");
			fprintf(of,"%lf\n",runtime);
			fclose(of);
		}
			//free the trees
		cudaFreeTree(&d_tree);
		freeTree(&h_tree);
	}
	return 0;
}
