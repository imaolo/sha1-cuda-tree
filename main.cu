#include <stdio.h>
#include "sha1.cu"

#define MESSAGE_SIZE 1
#define HASH_SIZE 20
#define MAX_ARITY 3

//use the current levels arity
#define getChildIdx(index,start,end,arity)\
	((index-start)*arity)+end+1;

//the arity is the parents arity 
#define getParentIdx(index,start,end,arity)\
	start - ((end-start+1)/arity - ((index - start)/arity))

typedef unsigned char UCHAR;
struct node {
	UCHAR hash[HASH_SIZE];
};
typedef struct node m_node;


void printHash(const UCHAR *hash){
	for (int i=0;i<HASH_SIZE;i++)
		printf("%02x",hash[i]);
	printf("\n");
}
void printTree(
	const m_node   *nodes,
	const uint64_t *startIdx, 
	const uint64_t *endIdx,
	uint8_t         height 
)
{

	printf("level: %d,",  height);
	printf("nodes: %ld\n", endIdx[height] - startIdx[height] + 1);
	for (uint64_t i=startIdx[height];i<=endIdx[height];i++)
		printHash(nodes[i].hash);
	printf("\n");
	if (height == 0)
		return;
	printTree(nodes,startIdx,endIdx,height-1);
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
	uint8_t thread = threadIdx.x;
	uint8_t block_size = blockDim.x;

	for (uint64_t idx=thread; idx<N; idx+=block_size){
		for (uint8_t i=0;i<arities[1];i++)
			SHA1((buffer+(i*HASH_SIZE)),message,MESSAGE_SIZE);
		SHA1(nodes[idx].hash,buffer,MESSAGE_SIZE*arities[1]);
		printf("leaf have been hashed\n");
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
			__syncthreads();

		}
	}

}

int main(int argc,char **argv){
	if (argc != 2){
		printf("enter correct args\n");
		return 0;
	}	

	//calculate the arities
	const uint64_t num_blocks = atoi(argv[1]);
	const uint8_t  height     = ceil(log10(num_blocks)/log10(3));
	const uint8_t  num_twos   = log10(num_blocks/pow(3,height))/log10(2.0f/3.0f);
	const uint8_t  num_threes = height - num_twos;
	const uint64_t num_leaves = pow(2,num_twos) * pow(3,num_threes);
	uint8_t  arities[height+1];
	arities[0] = 0;
	for (uint8_t i = 1;i<=num_threes;i++)
		arities[i] = 3;
	for (uint8_t i = num_threes+1;i<=height;i++)
		arities[i] = 2;
	//determine the offsets
	//they are is used to interleave addresses
	uint64_t offsets[height+1];
	offsets[1] = 1;
	for (uint8_t i=2;i<=height;i++)
		offsets[i] = arities[i]*offsets[i-1];
	//create the message string
	UCHAR message[MESSAGE_SIZE];
	for (int i=0;i<MESSAGE_SIZE;i++)
		message[i] = 'a';
	//create the nodes tree and initialize nodes
	m_node *nodes = (m_node*)malloc((num_leaves/arities[1])*sizeof(m_node));

	//allocate CudaMemory
	m_node   *d_nodes;
	UCHAR    *d_message;
	uint8_t  *d_arities;
	uint64_t *d_offsets;

	cudaMalloc(&d_message,MESSAGE_SIZE*sizeof(UCHAR));
	cudaMemcpy(d_message, message,MESSAGE_SIZE*sizeof(UCHAR),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_nodes,(num_leaves/arities[1])*sizeof(m_node));

	cudaMalloc(&d_arities,(height+1)*sizeof(uint8_t));
	cudaMemcpy(d_arities,arities,(height+1)*sizeof(uint8_t),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_offsets,(height+1)*sizeof(uint64_t));
	cudaMemcpy(d_offsets,offsets,(height+1)*sizeof(uint64_t),
		cudaMemcpyHostToDevice);

	//execute kernel function and extract the memory
	printf("N: %ld\n",num_leaves/arities[1]);
	printf("Invoking Kernel\n");
	hashTreeP<<<1, 1024>>>(
	 	d_nodes,
	 	num_leaves/arities[1],
	 	d_arities,
	 	d_offsets,
	 	height,
	 	d_message
	);
	cudaDeviceSynchronize();
	cudaMemcpy(nodes,d_nodes,(num_leaves/arities[1])*sizeof(m_node),
		cudaMemcpyDeviceToHost);
	
	printHash(nodes[0].hash);


	cudaFree(d_nodes);
	cudaFree(d_message);
	cudaFree(d_offsets);
	cudaFree(d_arities);
	free(nodes);
	return 0;
}
