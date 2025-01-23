#pragma once
#include <cub/cub.cuh>
#include "common.h"

namespace Aig {

__global__ void getElemTruthTable(unsigned * vTruthElem, int nVars);

__device__ __forceinline__
void cutTruthIter(int nodeId, const int * pFanin0, const int * pFanin1, 
                  unsigned * vTruthMem, int * visited, int * pVisitedSize, int nWords) {
    int fanin0Id, fanin1Id, fanin0TruthIdx, fanin1TruthIdx;
    int lit0, lit1;

    fanin0Id = dUtils::AigNodeID(pFanin0[nodeId]);
    fanin1Id = dUtils::AigNodeID(pFanin1[nodeId]);
    fanin0TruthIdx = fanin1TruthIdx = -1;
    for (int j = 0; j < *pVisitedSize; j++)
        if (visited[j] == fanin0Id) {
            fanin0TruthIdx = j;
            break;
        }
    for (int j = 0; j < *pVisitedSize; j++)
        if (visited[j] == fanin1Id) {
            fanin1TruthIdx = j;
            break;
        }
    assert(fanin0TruthIdx != -1 && fanin1TruthIdx != -1);

    lit0 = pFanin0[nodeId], lit1 = pFanin1[nodeId];
    if (!dUtils::AigNodeIsComplement(lit0) && !dUtils::AigNodeIsComplement(lit1))
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                vTruthMem[fanin0TruthIdx * nWords + i] & vTruthMem[fanin1TruthIdx * nWords + i];
    else if (!dUtils::AigNodeIsComplement(lit0) && dUtils::AigNodeIsComplement(lit1))
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                vTruthMem[fanin0TruthIdx * nWords + i] & ~vTruthMem[fanin1TruthIdx * nWords + i];
    else if (dUtils::AigNodeIsComplement(lit0) && !dUtils::AigNodeIsComplement(lit1))
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                ~vTruthMem[fanin0TruthIdx * nWords + i] & vTruthMem[fanin1TruthIdx * nWords + i];
    else
        for (int i = 0; i < nWords; i++)
            vTruthMem[(*pVisitedSize) * nWords + i] = 
                ~vTruthMem[fanin0TruthIdx * nWords + i] & ~vTruthMem[fanin1TruthIdx * nWords + i];

    visited[(*pVisitedSize)++] = nodeId;
}


/**
 * Device funtion of computing the truth table of one cut.
 * vTruthMem is the scratch memory used for intermediate node truth tables.
 * If it is not provided, it will be allocated dynamically.
 * vNumSaved is for debug uses in refactoring.
 */
template <int STACK_SIZE>
__device__ __forceinline__
void getCutTruthTableSingle(const int * pFanin0, const int * pFanin1, 
                            unsigned * pTruth, const unsigned * vTruthElem, 
                            const int * pCut, int nVars, int rootId, int nMaxCutSize, int nPIs,
                            unsigned * vTruthMem = NULL, const int * vNumSaved = NULL) {
    int nodeId;
    int fVisited, fAllocated, nWords, nWordsElem, nIntNodes;
    int stack[STACK_SIZE], stackRes[STACK_SIZE], visited[STACK_SIZE];
    int stackTop, stackResTop, visitedSize;

    nWords = dUtils::TruthWordNum(nVars);
    nWordsElem = dUtils::TruthWordNum(nMaxCutSize);

    visitedSize = 0;
    for (int i = 0; i < nVars; i++) {
        visited[visitedSize++] = pCut[i];
    }

    // 1. traversal to collect intermediate nodes
    stackTop = stackResTop = -1;
    stack[++stackTop] = rootId;
    while (stackTop != -1) {
        // skip if already visited
        nodeId = stack[stackTop--];
        fVisited = 0;
        for (int i = 0; i < visitedSize; i++) {
            if (visited[i] == nodeId) {
                fVisited = 1;
                break;
            }
        }
        if (fVisited) continue;

        assert(dUtils::AigIsNode(nodeId, nPIs));
        assert(visitedSize < STACK_SIZE);
        visited[visitedSize++] = nodeId;

        // save result. make sure the nodes in stackRes are in reversed topo order (decreasing id)
        int j = stackResTop++;
        for (; j >= 0 && stackRes[j] < nodeId; j--) // insertion sort
            stackRes[j + 1] = stackRes[j];
        stackRes[j + 1] = nodeId;

        // push fanins into stack
        assert(stackTop < STACK_SIZE - 2);
        stack[++stackTop] = dUtils::AigNodeID(pFanin1[nodeId]);
        stack[++stackTop] = dUtils::AigNodeID(pFanin0[nodeId]);
    }
    nIntNodes = stackResTop + 1;
    assert(stackRes[0] == rootId);
    // debug
    if (vNumSaved != NULL)
        assert(nIntNodes >= vNumSaved[rootId] && nIntNodes + nVars < STACK_SIZE);
    else
        assert(nIntNodes + nVars < STACK_SIZE);
    
    // 2. compute truth table
    // allocate memory for intermediate truth tables, if it is not provided
    if (vTruthMem == NULL) {
        vTruthMem = (unsigned *) malloc((nVars + nIntNodes) * nWords * sizeof(unsigned));
        assert(vTruthMem != NULL); // if NULL, then not enough heap memory
        fAllocated = 1;
    } else
        fAllocated = 0;
    
    visitedSize = 0;
    // collect elementary truth tables for the cut nodes
    for (int i = 0; i < nVars; i++) {
        for (int j = 0; j < nWords; j++)
            vTruthMem[i * nWords + j] = vTruthElem[i * nWordsElem + j];
        visited[visitedSize++] = pCut[i];
    }
    for (int i = stackResTop; i >= 0; i--) {
        cutTruthIter(stackRes[i], pFanin0, pFanin1, vTruthMem, visited, &visitedSize, nWords);
    }
    assert(visited[visitedSize - 1] == rootId);

    // copy the truth table of rootId to pTruth
    for (int i = 0; i < nWords; i++)
        pTruth[i] = vTruthMem[(visitedSize - 1) * nWords + i];
    
    if (fAllocated)
        free(vTruthMem);
}

/**
 * Get the truth tables of cuts rooted at vIndices, saved in table format.
 * The resulting truth tables are saved in the consecutive array vTruth. 
 **/
template <int CUT_TABLE_NUM_COLS, int STACK_SIZE>
__global__ void getCutTruthTable(const int * pFanin0, const int * pFanin1, const int * vNumSaved,
                                 const int * vIndices, const int * vCutTable, const int * vCutSizes, 
                                 unsigned * vTruth, const int * vTruthRanges, const unsigned * vTruthElem,
                                 int nIndices, int nPIs, int nMaxCutSize) {
    int nThreads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rootId, nVars;
    int startIdx;

    for (; idx < nIndices; idx += nThreads) {
        rootId = vIndices[idx];
        nVars = vCutSizes[rootId];
        startIdx = (idx == 0 ? 0 : vTruthRanges[idx - 1]);
        getCutTruthTableSingle<STACK_SIZE>(pFanin0, pFanin1, vTruth + startIdx, vTruthElem,
                                           vCutTable + rootId * CUT_TABLE_NUM_COLS, nVars,
                                           rootId, nMaxCutSize, nPIs, NULL, vNumSaved);
    }
}


/**
 * Same as getCutTruthTable, except that cuts are stored in a consecutive array.
 * Additionally, if vNode2ConeResynIdx is provided, mark the nodes in each cone using the 
 * corresponding cone index of vIndices.
 **/
template <int STACK_SIZE>
__global__ void getCutTruthTableConsecutive(const int * pFanin0, const int * pFanin1, const int * vNumSaved,
                                            const int * vIndices, const int * vCuts, const int * vCutRanges,
                                            unsigned * vTruth, const int * vTruthRanges, const unsigned * vTruthElem,
                                            int nIndices, int nPIs, int nMaxCutSize, int * vNode2ConeResynIdx = NULL) {
    typedef cub::WarpScan<int> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[THREAD_PER_BLOCK / 32];
    __shared__ unsigned * vTruthMemAlloc[THREAD_PER_BLOCK / 32];
    
    int nThreads = NUM_BLOCKS(nIndices, 32) * 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x / 32, laneIdx = threadIdx.x % 32;
    int stack[STACK_SIZE], stackRes[STACK_SIZE], visited[STACK_SIZE];
    int stackTop, stackResTop, visitedSize;
    int nodeId, rootId;
    int fVisited, nWords, nWordsElem, nVars, nIntNodes;
    unsigned * vTruthMem;
    int startIdx, endIdx;
    int localMemLen, totalMemLen = -1, startMemIdx = -1;

    assert(nIndices <= nThreads && nThreads - nIndices < 32);

    if (idx < nThreads) {
        if (idx < nIndices) {
            rootId = vIndices[idx];
            startIdx = (idx == 0 ? 0 : vCutRanges[idx - 1]);
            endIdx = vCutRanges[idx];
            nVars = endIdx - startIdx;
            nWords = dUtils::TruthWordNum(nVars);
            nWordsElem = dUtils::TruthWordNum(nMaxCutSize);

            // set the leaves as visited
            visitedSize = 0;
            for (int i = 0; i < nVars; i++) {
                visited[visitedSize++] = vCuts[startIdx + i];
            }

            // 1. traversal to collect intermediate nodes
            stackTop = stackResTop = -1;
            stack[++stackTop] = rootId;
            while (stackTop != -1) {
                // skip if already visited
                nodeId = stack[stackTop--];
                fVisited = 0;
                for (int i = 0; i < visitedSize; i++) {
                    if (visited[i] == nodeId) {
                        fVisited = 1;
                        break;
                    }
                }
                if (fVisited) continue;

                assert(dUtils::AigIsNode(nodeId, nPIs));
                assert(visitedSize < STACK_SIZE);
                visited[visitedSize++] = nodeId;

                // mark the node if vNode2ConeResynIdx is provided
                if (vNode2ConeResynIdx && nodeId != rootId) {
                    vNode2ConeResynIdx[nodeId] = idx;
                }

                // save result. make sure the nodes in stackRes are in reversed topo order (decreasing id)
                int j = stackResTop++;
                for (; j >= 0 && stackRes[j] < nodeId; j--) // insertion sort
                    stackRes[j + 1] = stackRes[j];
                stackRes[j + 1] = nodeId;

                // push fanins into stack
                assert(stackTop < STACK_SIZE - 2);
                stack[++stackTop] = dUtils::AigNodeID(pFanin1[nodeId]);
                stack[++stackTop] = dUtils::AigNodeID(pFanin0[nodeId]);
            }
            nIntNodes = stackResTop + 1;
            assert(nIntNodes >= vNumSaved[rootId] && nIntNodes + nVars < STACK_SIZE);
            assert(stackRes[0] == rootId);
        }

        // 2. compute truth table
        // allocate memory for intermediate truth tables
        // allocate once per warp to reduce overhead
        localMemLen = (idx < nIndices ? (nVars + nIntNodes) * nWords : 0);
        WarpScan(temp_storage[warpIdx]).ExclusiveSum(localMemLen, startMemIdx, totalMemLen);
        assert(startMemIdx != -1 && totalMemLen != -1);

        if (laneIdx == 0) {
            vTruthMemAlloc[warpIdx] = (unsigned *) malloc(totalMemLen * sizeof(unsigned));
            assert(vTruthMemAlloc[warpIdx] != NULL); // if NULL, then not enough heap memory
        }
        __syncwarp();
        vTruthMem = vTruthMemAlloc[warpIdx] + startMemIdx;
        

        if (idx < nIndices) {
            visitedSize = 0;
            // collect elementary truth tables for the cut nodes
            for (int i = 0; i < nVars; i++) {
                for (int j = 0; j < nWords; j++)
                    vTruthMem[i * nWords + j] = vTruthElem[i * nWordsElem + j];
                visited[visitedSize++] = vCuts[startIdx + i];
            }
            for (int i = stackResTop; i >= 0; i--) {
                cutTruthIter(stackRes[i], pFanin0, pFanin1, vTruthMem, visited, &visitedSize, nWords);
            }
            assert(visited[visitedSize - 1] == rootId);

            // copy the truth table of rootId to vTruth
            startIdx = (idx == 0 ? 0 : vTruthRanges[idx - 1]);
            endIdx = vTruthRanges[idx];
            assert(endIdx - startIdx == nWords);
            for (int i = 0; i < nWords; i++)
                vTruth[startIdx + i] = vTruthMem[(visitedSize - 1) * nWords + i];
        }

        __syncwarp();
        if (laneIdx == 0)
            free(vTruthMemAlloc[warpIdx]);
    }
}

} // namespace Aig


