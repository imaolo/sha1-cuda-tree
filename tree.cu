#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define HASH_SIZE 20

typedef struct
{
	unsigned char hash[HASH_SIZE];
} m_node;
typedef struct 
{
	m_node *nodes;
	uint8_t *arities;
	uint64_t *offsets;
	uint64_t *startIdx;
	uint64_t *endIdx;
	uint8_t height;
	unsigned char *message;
	uint64_t messageSize;
} m_tree;


//helper function for create*Tree functions
void configureTree(m_tree *tree)
{
	//fill offsets array
	tree->offsets[1] = 1;
	for (uint8_t i=2;i<=tree->height;i++)
		tree->offsets[i] = tree->arities[i]*tree->offsets[i-1];

	//fill startIdx and endIdx
	uint64_t nodes_at_level;
	for (uint64_t i = tree->height;i>=0; i--){
		if (i == tree->height){
			tree->startIdx[i] = 0;
			nodes_at_level = 1;
		}
		else{
			tree->startIdx[i] = tree->endIdx[i+1] + 1;
			nodes_at_level = (tree->endIdx[i+1] - tree->startIdx[i+1] + 1) *
			tree->arities[i+1];
		}
		tree->endIdx[i] = tree->startIdx[i] + nodes_at_level - 1;
		if (i == 0)
			break;
	}
}

//allocates memory for tree struct data
//computes tree struct's parameter data
void createOptimizedTree
(
	m_tree* tree,
	uint64_t numBlocks,
	uint64_t messageSize
)
{
	tree->messageSize  = messageSize;
	tree->height       = ceil(log10(numBlocks)/log10(3));
	uint8_t  numTwos   = log10(numBlocks/pow(3,tree->height))/log10(2.0f/3.0f);
	uint8_t  numThrees = tree->height - numTwos;
	tree->arities      = (uint8_t  *)malloc((tree->height+1)*sizeof(uint8_t));
	tree->offsets      = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->startIdx     = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->endIdx       = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->message      = (unsigned char *)malloc(messageSize*sizeof(unsigned char));
	
	//fill arities array
	tree->arities[0] = 0;
	for (uint8_t i = 1;i<=numThrees;i++)
		tree->arities[i] = 3;
	for (uint8_t i = numThrees+1;i<=tree->height;i++)
		tree->arities[i] = 2;

	//find offsets, startIDx, endIdx
	configureTree(tree);
	tree->nodes = (m_node *)malloc(sizeof(m_node)*(tree->endIdx[0]+1));
	
	//fill message string - arbitray message
	for (uint64_t i=0;i<messageSize;i++)
		tree->message[i] = 'a';
}
void createBinaryTree
(
	m_tree* tree,
	uint64_t numBlocks,
	uint64_t messageSize
)
{
	tree->messageSize = messageSize;
	tree->height      = ceil(log10(numBlocks)/log10(2));
	tree->arities     = (uint8_t  *)malloc((tree->height+1)*sizeof(uint8_t));
	tree->offsets     = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->startIdx    = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->endIdx      = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->message     = (unsigned char *)malloc(messageSize*sizeof(unsigned char));
	
	//fill arities array
	tree->arities[0] = 0;
	for (uint8_t i = 1;i<=tree->height;i++)
		tree->arities[i] = 2;

	//find offsets, startIDx, endIdx
	configureTree(tree);
	tree->nodes = (m_node *)malloc(sizeof(m_node)*(tree->endIdx[0]+1));

	//fill message string - arbitray message
	for (uint64_t i=0;i<messageSize;i++)
		tree->message[i] = 'a';
}


//memory management functions
void cudaCopyTree(m_tree *d_tree,m_tree *h_tree)
{
	d_tree->height       = h_tree->height;
	d_tree->messageSize  = h_tree->messageSize;
	cudaMalloc(&d_tree->message,d_tree->messageSize*sizeof(char));
	cudaMemcpy(
		d_tree->message, 
		h_tree->message,
		d_tree->messageSize*sizeof(char),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_tree->nodes,(h_tree->endIdx[1]-h_tree->startIdx[1]+1)*sizeof(m_node));

	cudaMalloc(&d_tree->arities,(h_tree->height+1)*sizeof(char));
	cudaMemcpy(
		d_tree->arities,
		h_tree->arities,
		(h_tree->height+1)*sizeof(uint8_t),
		cudaMemcpyHostToDevice
	);

	cudaMalloc(&d_tree->offsets,(h_tree->height+1)*sizeof(uint64_t));
	cudaMemcpy(
			d_tree->offsets,
			h_tree->offsets,
			(h_tree->height+1)*sizeof(uint64_t),
			cudaMemcpyHostToDevice
	);
}
void freeTree(m_tree *tree)
{
	free(tree->nodes);
	free(tree->arities);
	free(tree->offsets);
	free(tree->startIdx);
	free(tree->endIdx);
	free(tree->message);
}
void cudaFreeTree(m_tree *tree)
{
	cudaFree(tree->nodes);
	cudaFree(tree->message);
	cudaFree(tree->offsets);
	cudaFree(tree->arities);
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


//a serial merkle root generator for error checking
//the macros are used only in hashTreeS
//use the current levels arity
#define getChildIdx(index,start,end,arity)\
	((index-start)*arity)+end+1;
//the arity is the parents arity 
#define getParentIdx(index,start,end,arity)\
	start - ((end-start+1)/arity - ((index - start)/arity))
void hashTreeS (m_tree *tree)
{
	unsigned char *tmp = (unsigned char*)malloc(HASH_SIZE*3*sizeof(unsigned char));
	for (int i = tree->startIdx[0];i<=tree->endIdx[0];i++){
		SHA1(tree->nodes[i].hash,tree->message,tree->messageSize);
	}
	uint64_t childIdx;
	for (uint64_t i=1;i<=tree->height;i++){
		for (uint64_t j=tree->startIdx[i];j<=tree->endIdx[i];j++){
			childIdx = getChildIdx(j,tree->startIdx[i],tree->endIdx[i],tree->arities[i]);
			for (uint64_t k = childIdx;k<childIdx+tree->arities[i];k++)
				memcpy((tmp+(k-childIdx)*HASH_SIZE),tree->nodes[k].hash,HASH_SIZE);
			SHA1(tree->nodes[j].hash,tmp,HASH_SIZE*tree->arities[i]);
		}
	}
	free(tmp);
}

