#include "lbm.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include "lbmKernels.cuh"

#define THREADS_PER_BLOCK 8

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

lbmSolver::lbmSolver( uint nx, uint ny, float omega, float U, float V )
    : nx(nx), ny(ny), omega(omega), U(U), V(V)
{
    this->nx = nx;
    this->ny = ny;

    this->f.f00 = std::make_shared<floatVec>( nx * ny );
    this->f.fp0 = std::make_shared<floatVec>( nx * ny );
    this->f.fn0 = std::make_shared<floatVec>( nx * ny );
    this->f.fpp = std::make_shared<floatVec>( nx * ny );
    this->f.fnp = std::make_shared<floatVec>( nx * ny );
    this->f.fpn = std::make_shared<floatVec>( nx * ny );
    this->f.fnn = std::make_shared<floatVec>( nx * ny );
    this->f.f0p = std::make_shared<floatVec>( nx * ny );
    this->f.f0n = std::make_shared<floatVec>( nx * ny );

    this->f.geo = std::make_shared<charVec> ( nx * ny );

    this->f.pressure = std::make_shared<floatVec>( nx * ny );
    this->f.velocity = std::make_shared<floatVec>( nx * ny );

    this->minPressure = -1.0e-3f;
    this->maxPressure =  1.0e-3f;

    this->minVelocity =  0.0;
    this->maxVelocity =  0.02;
}

lbmSolver::~lbmSolver()
{
}

void lbmSolver::connectVertexBuffer(uint vertexBufferID)
{
    // register the OpenGL vertex Buffer within CUDA
    checkCudaErrors( cudaGraphicsGLRegisterBuffer(&this->glVertexBufferResource, 
                                                  vertexBufferID, 
                                                  cudaGraphicsMapFlagsNone) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void lbmSolver::initializeDistributions()
{
    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    initializeDistributionsKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny, this->U, this->V );

    swap( f.f0n, f.f0p );
    swap( f.fnn, f.fpp );
    swap( f.fp0, f.fn0 );
    swap( f.fpn, f.fnp );

    initializeDistributionsKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny, this->U, this->V );

    scaleColorMap();
}

void lbmSolver::collision()
{
    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    //////////////////////////////////////////////////////////////////////////

    collisionKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny, this->omega, this->U, this->V );
    getLastCudaError("collisionKernel failed.");

    //////////////////////////////////////////////////////////////////////////

    swap( f.f0n, f.f0p );
    swap( f.fnn, f.fpp );
    swap( f.fp0, f.fn0 );
    swap( f.fpn, f.fnp );
}

void lbmSolver::postProcessing( char type )
{
    this->computeMacroscopicQuantities();

    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    cudaGraphicsMapResources(1, &this->glVertexBufferResource, 0);
    getLastCudaError("cudaGraphicsMapResources failed");

    float* verticesDev;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&verticesDev, &num_bytes, this->glVertexBufferResource);
    getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

    float min;
    float max;

    if( type == 'p' ){
        min = this->minPressure;
        max = this->maxPressure;
    }
    else{
        min = this->minVelocity;
        max = this->maxVelocity;
    }

    postProcessingSetColorKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny, verticesDev, type, min, max );
    getLastCudaError("postProcessingSetColorKernel failed.");

    cudaGraphicsUnmapResources(1, &this->glVertexBufferResource, 0);
    getLastCudaError("cudaGraphicsUnmapResources failed");
}

void lbmSolver::computeMacroscopicQuantities()
{
    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    postProcessingMacroscopicQuantitiesKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny );
    getLastCudaError("postProcessingMacroscopicQuantitiesKernel failed.");
}

void lbmSolver::scaleColorMap()
{
    this->computeMacroscopicQuantities();

    this->minPressure = thrust::reduce( this->f.pressure->begin(), this->f.pressure->end(),  10.0f, thrust::minimum<float>() );
    this->maxPressure = thrust::reduce( this->f.pressure->begin(), this->f.pressure->end(), -10.0f, thrust::maximum<float>() );

    std::cout << "Pressure = ( " << this->minPressure << ", " << this->maxPressure << " )" << std::endl;

    this->minVelocity = thrust::reduce( this->f.velocity->begin(), this->f.velocity->end(),  10.0f, thrust::minimum<float>() );
    this->maxVelocity = thrust::reduce( this->f.velocity->begin(), this->f.velocity->end(), -10.0f, thrust::maximum<float>() );

    std::cout << "Velocity = ( " << this->minVelocity << ", " << this->maxVelocity << " )" << std::endl;
}

void lbmSolver::setGeo(uint xIdx, uint yIdx, char geo)
{
    dim3 threads ( 2, 2 );

    if( geo == 0 ){
        threads.x += 4;
        threads.y += 4;
    }

    setGeoKernel<<<1, threads>>>( this->getDistPtr(), this->nx, this->ny, xIdx, yIdx, geo );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void lbmSolver::swap( floatVecPtr& lhs, floatVecPtr& rhs )
{
    floatVecPtr tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

uint lbmSolver::c2i( uint xIdx, uint yIdx )
{
    return yIdx * this->nx + xIdx;
}

D2Q9Ptr lbmSolver::getDistPtr()
{
    D2Q9Ptr distPtr;

    distPtr.f00 = this->f.f00->data();
    distPtr.fp0 = this->f.fp0->data();
    distPtr.fn0 = this->f.fn0->data();
    distPtr.fpp = this->f.fpp->data();
    distPtr.fnp = this->f.fnp->data();
    distPtr.fpn = this->f.fpn->data();
    distPtr.fnn = this->f.fnn->data();
    distPtr.f0p = this->f.f0p->data();
    distPtr.f0n = this->f.f0n->data();
    distPtr.geo = this->f.geo->data();

    distPtr.velocity = this->f.velocity->data();
    distPtr.pressure = this->f.pressure->data();

    return distPtr;
}
