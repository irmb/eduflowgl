#include "lbm.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

#define THREADS_PER_BLOCK 8

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initializeDistributionsKernel( D2Q9 f, uint nx, uint ny )
{
    uint xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( xIdx >= nx - 1 || yIdx >= ny - 1 ) return;

    uint nodeIdx = yIdx * nx + xIdx;

    float U = 0.0f;
    float V = 0.0f;

    f.f00[ nodeIdx ] = (   (-2       + 3*(U*U))*(-2       + 3*(V*V)))/9.  - 4.0f/9.0f  ;
    f.fp0[ nodeIdx ] = ( - ( 1 + 3*U + 3*(U*U))*(-2       + 3*(V*V)))/18. - 1.0f/9.0f  ;
    f.fn0[ nodeIdx ] = ( - ( 1 - 3*U + 3*(U*U))*(-2       + 3*(V*V)))/18. - 1.0f/9.0f  ;
    f.f0p[ nodeIdx ] = ( - (-2       + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/18. - 1.0f/9.0f  ;
    f.f0n[ nodeIdx ] = ( - (-2       + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/18. - 1.0f/9.0f  ;
    f.fpp[ nodeIdx ] = (   ( 1 + 3*U + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/36. - 1.0f/36.0f ;
    f.fpn[ nodeIdx ] = (   ( 1 + 3*U + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/36. - 1.0f/36.0f ;
    f.fnn[ nodeIdx ] = (   ( 1 - 3*U + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/36. - 1.0f/36.0f ;
    f.fnp[ nodeIdx ] = (   ( 1 - 3*U + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/36. - 1.0f/36.0f ;

    //if      ( xIdx == nx - 1 || yIdx == ny - 1 ) f.geo[ nodeIdx ] = 1;
    //else if ( xIdx == 0      || xIdx == nx - 2 ) f.geo[ nodeIdx ] = 2;
    //else if ( yIdx == 0      || yIdx == ny - 2 ) f.geo[ nodeIdx ] = 3;
    //else                                         f.geo[ nodeIdx ] = 0;

    if      ( xIdx == nx - 1 || yIdx == ny - 1 ) f.geo[ nodeIdx ] = 1;
    else if ( yIdx == 0      || xIdx == 0      || xIdx == nx - 2 ) f.geo[ nodeIdx ] = 2;
    else if ( yIdx == ny - 2 ) f.geo[ nodeIdx ] = 3;
    else                                         f.geo[ nodeIdx ] = 0;

}

__global__ void collisionKernel( D2Q9 f, uint nx, uint ny, float omega, float* vertices )
{
    uint xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( xIdx >= nx - 1 || yIdx >= ny - 1 ) return;

    uint nodeIdx00 = ( xIdx     ) + ( yIdx     ) * nx;

    uint nodeIdxp0 = ( xIdx + 1 ) + ( yIdx     ) * nx;
    uint nodeIdxpp = ( xIdx + 1 ) + ( yIdx + 1 ) * nx;
    uint nodeIdx0p = ( xIdx     ) + ( yIdx + 1 ) * nx;

    //////////////////////////////////////////////////////////////////////////

    char  geo = f.geo[ nodeIdx00 ];

    if( geo == 1 ){
        vertices[ 5 * nodeIdx00 + 2 ] = 0.0f;
        vertices[ 5 * nodeIdx00 + 4 ] = 0.0f;
        return;
    }

    float g00 = f.f00[ nodeIdx00 ];
    float g0p = f.f0p[ nodeIdx00 ];
    float gpp = f.fpp[ nodeIdx00 ];
    float gp0 = f.fp0[ nodeIdx00 ];

    float gn0 = f.fn0[ nodeIdxp0 ];
    float gnp = f.fnp[ nodeIdxp0 ];

    float gpn = f.fpn[ nodeIdx0p ];
    float g0n = f.f0n[ nodeIdx0p ];

    float gnn = f.fnn[ nodeIdxpp ];

    //////////////////////////////////////////////////////////////////////////

    float dRho = ( ( ( gnp + gpn ) + ( gnn + gpp ) ) + ( ( g0p + g0n ) + ( gp0 + gn0 ) ) ) + g00;

    float U   = ( ( ( - gnp + gpn ) + ( - gnn + gpp ) ) + (                 ( gp0 - gn0 ) ) ) / ( 1.0f + dRho );
    float V   = ( ( (   gnp - gpn ) + ( - gnn + gpp ) ) + ( ( g0p - g0n )                 ) ) / ( 1.0f + dRho );

    //////////////////////////////////////////////////////////////////////////

    if (nodeIdx00==1050){
    float a=1.f;
    }

    if( geo == 0 ){
        g00 = g00 * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * (-2       + 3*(U*U))*(-2       + 3*(V*V)))/9.  - 4.0f/9.0f  );
        gp0 = gp0 * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * ( 1 + 3*U + 3*(U*U))*(-2       + 3*(V*V)))/18. - 1.0f/9.0f  );
        gn0 = gn0 * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * ( 1 - 3*U + 3*(U*U))*(-2       + 3*(V*V)))/18. - 1.0f/9.0f  );
        g0p = g0p * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * (-2       + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/18. - 1.0f/9.0f  );
        g0n = g0n * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * (-2       + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/18. - 1.0f/9.0f  );
        gpp = gpp * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 + 3*U + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
        gpn = gpn * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 + 3*U + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
        gnn = gnn * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 - 3*U + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
        gnp = gnp * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 - 3*U + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
    }
    else if( geo == 2 ){
        V = 0.0f;
        U = 0.01f;

        g00 = (   ( (-2 + 3*(U*U))*(-2 + 3*(V*V)))/9. -4.0f/9.0f );
        gp0 = ( - ( (1 + 3*U + 3*(U*U))*(-2 + 3*(V*V)))/18. -1.0f/9.0f);
        gn0 = ( - ( (1 - 3*U + 3*(U*U))*(-2 + 3*(V*V)))/18. -1.0f/9.0f);
        g0p = ( - ( (-2 + 3*(U*U))*(1 + 3*V + 3*(V*V)))/18. -1.0f/9.0f);
        g0n = ( - ( (-2 + 3*(U*U))*(1 - 3*V + 3*(V*V)))/18. -1.0f/9.0f);
        gpp = (   ( (1 + 3*U + 3*(U*U))*(1 + 3*V + 3*(V*V)))/36. -1.0f/36.0f);
        gpn = (   ( (1 + 3*U + 3*(U*U))*(1 - 3*V + 3*(V*V)))/36. -1.0f/36.0f);
        gnn = (   ( (1 - 3*U + 3*(U*U))*(1 - 3*V + 3*(V*V)))/36. -1.0f/36.0f);
        gnp = (   ( (1 - 3*U + 3*(U*U))*(1 + 3*V + 3*(V*V)))/36. -1.0f/36.0f);
    }
    else if( geo == 3 ){
        V = 0.0f;
        U = 0.01f;

        g00 = (   ( (-2 + 3*(U*U))*(-2 + 3*(V*V)))/9. -4.0f/9.0f );
        gp0 = ( - ( (1 + 3*U + 3*(U*U))*(-2 + 3*(V*V)))/18. -1.0f/9.0f);
        gn0 = ( - ( (1 - 3*U + 3*(U*U))*(-2 + 3*(V*V)))/18. -1.0f/9.0f);
        g0p = ( - ( (-2 + 3*(U*U))*(1 + 3*V + 3*(V*V)))/18. -1.0f/9.0f);
        g0n = ( - ( (-2 + 3*(U*U))*(1 - 3*V + 3*(V*V)))/18. -1.0f/9.0f);
        gpp = (   ( (1 + 3*U + 3*(U*U))*(1 + 3*V + 3*(V*V)))/36. -1.0f/36.0f);
        gpn = (   ( (1 + 3*U + 3*(U*U))*(1 - 3*V + 3*(V*V)))/36. -1.0f/36.0f);
        gnn = (   ( (1 - 3*U + 3*(U*U))*(1 - 3*V + 3*(V*V)))/36. -1.0f/36.0f);
        gnp = (   ( (1 - 3*U + 3*(U*U))*(1 + 3*V + 3*(V*V)))/36. -1.0f/36.0f);
    }

    //////////////////////////////////////////////////////////////////////////

    f.f00[ nodeIdx00 ] = g00;
    f.f0p[ nodeIdx00 ] = g0n;
    f.fpp[ nodeIdx00 ] = gnn;
    f.fp0[ nodeIdx00 ] = gn0;

    f.fn0[ nodeIdxp0 ] = gp0;
    f.fnp[ nodeIdxp0 ] = gpn;

    f.fpn[ nodeIdx0p ] = gnp;
    f.f0n[ nodeIdx0p ] = g0p;

    f.fnn[ nodeIdxpp ] = gpp;

    //////////////////////////////////////////////////////////////////////////
    
    vertices[ 5 * nodeIdx00 + 2 ] =        sqrt( U*U + V*V ) / 0.02f;
    vertices[ 5 * nodeIdx00 + 4 ] = 1.0f - sqrt( U*U + V*V ) / 0.02f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

lbmSolver::lbmSolver(uint nx, uint ny)
    : nx(nx), ny(ny)
{
    this->nx = nx;
    this->ny = ny;

    checkCudaErrors( cudaMalloc( (void **) &this->f.f00 , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.fp0 , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.fn0 , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.fpp , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.fnp , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.fpn , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.fnn , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.f0p , nx * ny * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void **) &this->f.f0n , nx * ny * sizeof(float) ) );

    checkCudaErrors( cudaMalloc( (void **) &this->f.geo , nx * ny * sizeof(char) ) );

    this->f.geoHost = new char [ nx * ny ];
}

lbmSolver::~lbmSolver()
{
    checkCudaErrors( cudaFree( (void **) &this->f.f00 ) );
    checkCudaErrors( cudaFree( (void **) &this->f.fp0 ) );
    checkCudaErrors( cudaFree( (void **) &this->f.fn0 ) );
    checkCudaErrors( cudaFree( (void **) &this->f.fpp ) );
    checkCudaErrors( cudaFree( (void **) &this->f.fnp ) );
    checkCudaErrors( cudaFree( (void **) &this->f.fpn ) );
    checkCudaErrors( cudaFree( (void **) &this->f.fnn ) );
    checkCudaErrors( cudaFree( (void **) &this->f.f0p ) );
    checkCudaErrors( cudaFree( (void **) &this->f.f0n ) );

    checkCudaErrors( cudaFree( (void **) &this->f.geo ) );

    delete [] this->f.geoHost;
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

void lbmSolver::collision()
{
    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    //////////////////////////////////////////////////////////////////////////

    cudaGraphicsMapResources(1, &this->glVertexBufferResource, 0);
    getLastCudaError("cudaGraphicsMapResources failed");

    float* verticesDev;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&verticesDev, &num_bytes, this->glVertexBufferResource);
    getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

    collisionKernel<<<blocks, threads>>>( this->f, this->nx, this->ny, 1.99f, verticesDev );
    getLastCudaError("colorKernel failed.");

    cudaGraphicsUnmapResources(1, &this->glVertexBufferResource, 0);
    getLastCudaError("cudaGraphicsUnmapResources failed");

    //////////////////////////////////////////////////////////////////////////

    swap( &f.f0n, &f.f0p );
    swap( &f.fnn, &f.fpp );
    swap( &f.fp0, &f.fn0 );
    swap( &f.fpn, &f.fnp );
}

void lbmSolver::swap( float** lhs, float** rhs )
{
    float* tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}

void lbmSolver::setSolid(uint xIdx, uint yIdx)
{
    if( c2i( xIdx    , yIdx     ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx    , yIdx     ) ] = 1;
    if( c2i( xIdx + 1, yIdx     ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx + 1, yIdx     ) ] = 1;
    if( c2i( xIdx + 1, yIdx + 1 ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx + 1, yIdx + 1 ) ] = 1;
    if( c2i( xIdx    , yIdx + 1 ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx    , yIdx + 1 ) ] = 1;
}

void lbmSolver::setFluid(uint xIdx, uint yIdx)
{
    if( c2i( xIdx    , yIdx     ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx    , yIdx     ) ] = 0;
    if( c2i( xIdx + 1, yIdx     ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx + 1, yIdx     ) ] = 0;
    if( c2i( xIdx + 1, yIdx + 1 ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx + 1, yIdx + 1 ) ] = 0;
    if( c2i( xIdx    , yIdx + 1 ) < this->nx * this->ny ) this->f.geoHost[ c2i( xIdx    , yIdx + 1 ) ] = 0;
}

void lbmSolver::uploadGeo()
{
    checkCudaErrors( cudaMemcpy( this->f.geo, this->f.geoHost, this->nx * this->ny * sizeof(char), cudaMemcpyHostToDevice ) );
}

void lbmSolver::downloadGeo()
{
    checkCudaErrors( cudaMemcpy( this->f.geoHost, this->f.geo, this->nx * this->ny * sizeof(char), cudaMemcpyDeviceToHost ) );
}

void lbmSolver::initializeDistributions()
{

    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    initializeDistributionsKernel<<<blocks, threads>>>( this->f, this->nx, this->ny );

    swap( &f.f0n, &f.f0p );
    swap( &f.fnn, &f.fpp );
    swap( &f.fp0, &f.fn0 );
    swap( &f.fpn, &f.fnp );

    initializeDistributionsKernel<<<blocks, threads>>>( this->f, this->nx, this->ny );

    downloadGeo();
}

uint lbmSolver::c2i( uint xIdx, uint yIdx )
{
    return yIdx * this->nx + xIdx;
}