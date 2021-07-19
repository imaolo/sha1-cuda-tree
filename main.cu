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
	uint8_t hashed;
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
	uint64_t *startIdx,
	uint64_t *endIdx,
	uint8_t  *arities,
	uint8_t  height,
	const UCHAR    *message
)
{
	//create a buffer that will be used several times;
	UCHAR buffer[HASH_SIZE*MAX_ARITY];
	//find location and set currIdx and childIdx
	uint64_t curr = blockIdx.x * blockDim.x + threadIdx.x;
	curr += startIdx[1];
	if (curr>endIdx[1])
		return;
	uint64_t childIdx = getChildIdx(curr,startIdx[1],endIdx[1],arities[1]);
	//hash children and store the concatenated results in the buff
	for (uint8_t i = 0;i<arities[1];i++){
		SHA1((buffer+(i*HASH_SIZE)),message,MESSAGE_SIZE);
		memcpy(nodes[childIdx+i].hash,(buffer+(i*HASH_SIZE)),HASH_SIZE);
		nodes[childIdx+i].hashed = 1;
	}
	//hash the concatenations together
	SHA1(nodes[curr].hash,buffer,HASH_SIZE*arities[1]);
	nodes[curr].hashed = 1;
	//only one sibling moves to the parent
	if (curr%arities[2] != 0)
		return;

	//go to the parent node, now at level 2, save the child index
	curr = getParentIdx(curr,startIdx[1],endIdx[1],arities[2]);

	//iterate through the tree
	for (uint8_t i=2;i<=height;i++){
		childIdx = getChildIdx(curr,startIdx[i],endIdx[i],arities[i]);
		//wait for the children to be hashed
		uint8_t flag;
		while(1){
			flag = 0;
			for (uint8_t j=0;j<arities[i];j++){
				if (nodes[childIdx+j].hashed==1)
					flag++;
			}
			if(flag == arities[i])
				break;
		}
		for (int j=0;j<arities[i];j++){
			if (nodes[childIdx+j].hashed == 0)
				printf("oh no\n");
		}
		//concat the children
		for (uint8_t j=0;j<arities[i];j++)
			memcpy((buffer+(j*HASH_SIZE)),nodes[childIdx+j].hash,HASH_SIZE);
		//hash the concatenations
		SHA1(nodes[curr].hash,buffer,HASH_SIZE*arities[i]);
		//only one sibling continues
		if (curr == 0  | curr%arities[i+1] != 0)
			return;
		//move onto next level
		curr = getParentIdx(curr,startIdx[i],endIdx[i],arities[i+1]);
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

	//determine start and end index for each level
	//they are is used to navigate the tree
	uint64_t startIdx[height+1];
	uint64_t endIdx[height+1];
	uint64_t nodes_at_level;
	for (uint64_t i = height;i>=0; i--){
		if (i == height){
			startIdx[i] = 0;
			nodes_at_level = 1;
		}
		else{
			startIdx[i] = endIdx[i+1] + 1;
			nodes_at_level = (endIdx[i+1] - startIdx[i+1] + 1 ) * arities[i+1];
		}
		endIdx[i] = startIdx[i] + nodes_at_level - 1;
		if (i == 0)
			break;
	}


	//create the message string
	UCHAR message[MESSAGE_SIZE];
	for (int i=0;i<MESSAGE_SIZE;i++)
		message[i] = 'a';
	//create the nodes tree
	m_node *nodes = (m_node*)malloc((endIdx[0]+1)*sizeof(m_node));
	for (uint64_t i =0;i<=endIdx[0];i++)
		nodes[i].hashed = 0;

	//allocate CudaMemory
	m_node   *d_nodes;
	UCHAR    *d_message;
	uint64_t *d_startIdx,*d_endIdx;
	uint8_t  *d_arities;

	cudaMalloc(&d_message,MESSAGE_SIZE*sizeof(UCHAR));
	cudaMemcpy(d_message, message,MESSAGE_SIZE*sizeof(UCHAR),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_nodes,(endIdx[0]+1)*sizeof(m_node));

	cudaMalloc(&d_startIdx,(height+1)*sizeof(uint64_t));
	cudaMemcpy(d_startIdx, startIdx,(height+1)*sizeof(uint64_t),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_endIdx,(height+1)*sizeof(uint64_t));
	cudaMemcpy(d_endIdx, endIdx,(height+1)*sizeof(uint64_t),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_arities,(height+1)*sizeof(uint8_t));
	cudaMemcpy(d_arities,arities,(height+1)*sizeof(uint8_t),
		cudaMemcpyHostToDevice);

	//execute kernel function and extract the memory
	printf("Kernel Working... \n");
	uint64_t N = endIdx[1] - startIdx[1] + 1;
	hashTreeP<<< ( (N+255)/256 ) , 256 >>>(
	 	d_nodes,
	 	d_startIdx,
	 	d_endIdx,
	 	d_arities,
	 	height,
	 	d_message
	);
	cudaDeviceSynchronize();
	cudaMemcpy(nodes,d_nodes,(endIdx[0]+1)*sizeof(m_node),
		cudaMemcpyDeviceToHost);
	
	printTree(nodes,startIdx,endIdx,height);

	cudaFree(d_nodes);
	cudaFree(d_message);
	cudaFree(d_endIdx);
	cudaFree(d_startIdx);
	cudaFree(d_arities);
	free(nodes);
	return 0;
}
