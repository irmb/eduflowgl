#include "cudaKernel.h"

cudaGraphicsResource* glVertexBufferResource;

__global__ void postProcessingSetColorKernel( float* vertices, float delta )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    vertices[ 6 * idx + 0 ] = vertices[ 6 * idx + 0 ] + delta;
}

void changeTriangle( float delta )
{
    cudaGraphicsMapResources(1, &glVertexBufferResource, 0);

    float* verticesDev;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&verticesDev, &num_bytes, glVertexBufferResource);

    postProcessingSetColorKernel<<<1, 3>>>(verticesDev, delta);

    cudaGraphicsUnmapResources(1, &glVertexBufferResource, 0);
}

void connectVertexBuffer(unsigned int vertexBufferID)
{
    // register the OpenGL vertex Buffer within CUDA
    cudaGraphicsGLRegisterBuffer( &glVertexBufferResource, 
                                  vertexBufferID, 
                                  cudaGraphicsMapFlagsNone );
}