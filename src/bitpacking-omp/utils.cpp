#include "utils.h"

size_t getReduceScratchSpaceSize(size_t const num)
{
  size_t const base =
      std::min(BLOCK_WIDTH, static_cast<int>(roundUpDiv(num, (size_t)BLOCK_SIZE)))
      * sizeof(uint64_t);
  return base;
}

size_t requiredWorkspaceSize(size_t const num, const nvcompType_t type)
{
  return sizeOfnvcompType(type) * getReduceScratchSpaceSize(num) * 2;
}
