#ifndef lbmKernels_H
#define lbmKernels_H

struct D2Q9Ptr;

typedef unsigned int uint;

__global__ void initializeDistributionsKernel(D2Q9Ptr f, uint nx, uint ny, float U, float V);

__global__ void initializeGeoKernel(D2Q9Ptr f, uint nx, uint ny);

__global__ void collisionKernel(D2Q9Ptr f, uint nx, uint ny, float omega, float U, float V, char model);

__global__ void postProcessingMacroscopicQuantitiesKernel(D2Q9Ptr f, uint nx, uint ny);

__global__ void postProcessingSetColorKernel(D2Q9Ptr f, uint nx, uint ny, float* vertices, char type, float min, float max, char geoMode);

__global__ void setGeoKernel(D2Q9Ptr f, uint nx, uint ny, uint x, uint y, char geo);

#endif