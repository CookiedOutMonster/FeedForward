#ifndef GPUCONFIG_H
#define GPUCONFIG_H

extern int THREADS_PER_BLOCK;
void initGPUConfig();
int calculateBlocks(int size);

#endif  // CONSTANTS_H