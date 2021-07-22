#include <stdio.h>
#include <time.h>
#include "sha1.cu"
#include "tree.cu"

#define MESSAGE_SIZE 1
#define HASH_SIZE 20
#define MAX_ARITY 3

typedef unsigned char UCHAR;


void printHash(const UCHAR *hash){
	for (int i=0;i<HASH_SIZE;i++)
		printf("%02x",hash[i]);
	printf("\n");
}


__global__ 
void hashTreeP 
(
	m_node   *nodes,
	uint64_t N,
	uint8_t  *arities,
	uint64_t *offsets,
	uint8_t  height,
	const UCHAR    *message
)
{
	UCHAR buffer[HASH_SIZE*MAX_ARITY];
	uint16_t thread = threadIdx.x;
	uint16_t block_size = blockDim.x;
	for (uint64_t idx=thread; idx<N; idx+=block_size){
		for (uint8_t i=0;i<arities[1];i++)
			SHA1((buffer+(i*HASH_SIZE)),message,MESSAGE_SIZE);
		SHA1(nodes[idx].hash,buffer,HASH_SIZE*arities[1]);
	}
	__syncthreads();

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

int main(int argc,char **argv){
	if (argc != 2){
		printf("enter correct args\n");
		return 0;
	}	

	//configure tree
	m_tree h_tree;
	m_tree d_tree;
	configureBinaryTree(&h_tree,atoi(argv[1]),MESSAGE_SIZE);
	h_tree.nodes = (m_node *)malloc(sizeof(m_node)*(h_tree.endIdx[0]+1));

	//allocate cuda memory
	d_tree.height = h_tree.height;
	d_tree.messageSize  = h_tree.messageSize;
	cudaMalloc(&d_tree.message,MESSAGE_SIZE*sizeof(UCHAR));
	cudaMemcpy(
		d_tree.message, 
		h_tree.message,
		MESSAGE_SIZE*sizeof(UCHAR),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_tree.nodes,(h_tree.endIdx[1]-h_tree.startIdx[1]+1)*sizeof(m_node));

	cudaMalloc(&d_tree.arities,(h_tree.height+1)*sizeof(uint8_t));
	cudaMemcpy(
		d_tree.arities,
		h_tree.arities,
		(h_tree.height+1)*sizeof(uint8_t),
		cudaMemcpyHostToDevice
	);

	cudaMalloc(&d_tree.offsets,(h_tree.height+1)*sizeof(uint64_t));
	cudaMemcpy(
			d_tree.offsets,
			h_tree.offsets,
			(h_tree.height+1)*sizeof(uint64_t),
			cudaMemcpyHostToDevice
	);

	//execute kernel function and extract the memory
	uint64_t N = h_tree.endIdx[1] - h_tree.startIdx[1] + 1;
	printf("Invoking Kernel\n");
	double start = clock();
	hashTreeP<<<1, 1024>>>(
	 	d_tree.nodes,
	 	N,
	 	d_tree.arities,
	 	d_tree.offsets,
	 	d_tree.height,
	 	d_tree.message
	);
	cudaDeviceSynchronize();
	printf("gpu seconds: %4lf\n",(clock()-start)/CLOCKS_PER_SEC);
	unsigned char d_merkle_root[HASH_SIZE];
	cudaMemcpy(
		d_merkle_root,
		d_tree.nodes[0].hash,
		HASH_SIZE*sizeof(unsigned char),
		cudaMemcpyDeviceToHost);

	start = clock();
	hashTreeS(&h_tree);
	printf("cpu seconds: %4lf\n",(clock()-start)/CLOCKS_PER_SEC);

	printHash(d_merkle_root);
	printHash(h_tree.nodes[0].hash);


	cudaFree(d_tree.nodes);
	cudaFree(d_tree.message);
	cudaFree(d_tree.offsets);
	cudaFree(d_tree.arities);
	freeTree(&h_tree);
	return 0;
}
