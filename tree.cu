#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define HASH_SIZE 20

//use the current levels arity
#define getChildIdx(index,start,end,arity)\
	((index-start)*arity)+end+1;

//the arity is the parents arity 
#define getParentIdx(index,start,end,arity)\
	start - ((end-start+1)/arity - ((index - start)/arity))

typedef struct {
	unsigned char hash[HASH_SIZE];
} m_node;
typedef struct {
	m_node *nodes;
	uint8_t *arities;
	uint64_t *offsets;
	uint64_t *startIdx;
	uint64_t *endIdx;
	uint8_t height;
	unsigned char *message;
	uint64_t messageSize;
} m_tree;



//does not allocate space for the nodes, that is algorithm specific
void configureOptimizedTree(
	m_tree* tree,
	uint64_t numBlocks,
	uint64_t messageSize
)
{
	tree->messageSize   = messageSize;
	tree->height        = ceil(log10(numBlocks)/log10(3));
	uint8_t  num_twos   = log10(numBlocks/pow(3,tree->height))/log10(2.0f/3.0f);
	uint8_t  num_threes = tree->height - num_twos;
	uint64_t num_leaves = pow(2,num_twos) * pow(3,num_threes);
	tree->arities  = (uint8_t  *)malloc((tree->height+1)*sizeof(uint8_t));
	tree->offsets  = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->startIdx = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->endIdx   = (uint64_t *)malloc((tree->height+1)*sizeof(uint64_t));
	tree->message  = (unsigned char *)malloc(messageSize*sizeof(unsigned char));
	
	//fill arities array
	tree->arities[0] = 0;
	for (uint8_t i = 1;i<=num_threes;i++)
		tree->arities[i] = 3;
	for (uint8_t i = num_threes+1;i<=tree->height;i++)
		tree->arities[i] = 2;

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

	//fill message string - arbitray message
	for (uint64_t i=0;i<messageSize;i++)
		tree->message[i] = 'a';
}

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

void freeTree(m_tree *tree)
{
	free(tree->nodes);
	free(tree->arities);
	free(tree->offsets);
	free(tree->startIdx);
	free(tree->endIdx);
	free(tree->message);
}

void Print(const unsigned char *message, uint16_t n)
{
	for (int i=0;i<n;i++)
		printf("%02x",message[i]);
	printf("\n");
}
void printTree(m_tree *tree)
{
	for (uint8_t i=tree->height;i>=0;i--){
		printf("level: %d\n", i);
		printf("nodes: %ld\n", tree->endIdx[i] - tree->startIdx[i] + 1);
		for (uint64_t j=tree->startIdx[i];j<=tree->endIdx[i];j++)
			Print(tree->nodes[j].hash,HASH_SIZE);
		printf("\n");
		if (i == 0)
			return;
	}
}

