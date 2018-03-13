#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

// always include glew.h before any other openGL stuff
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern cudaGraphicsResource* glVertexBufferResource;

void changeTriangle( float delta );

void connectVertexBuffer(unsigned int vertexBufferID);

#endif