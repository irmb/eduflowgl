#include "cudaKernel.h"

cudaGraphicsResource* glVertexBufferResource;

__global__ void postProcessingSetColorKernel( float* vertices )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    vertices[ 6 * idx + 0 ] = vertices[ 6 * idx + 0 ] + 0.001f;
}

void changeTriangle()
{
    cudaGraphicsMapResources(1, &glVertexBufferResource, 0);

    float* verticesDev;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&verticesDev, &num_bytes, glVertexBufferResource);

    postProcessingSetColorKernel<<<1, 3>>>(verticesDev);

    cudaGraphicsUnmapResources(1, &glVertexBufferResource, 0);
}

void connectVertexBuffer(unsigned int vertexBufferID)
{
    // register the OpenGL vertex Buffer within CUDA
    cudaGraphicsGLRegisterBuffer( &glVertexBufferResource, 
                                  vertexBufferID, 
                                  cudaGraphicsMapFlagsNone );
}