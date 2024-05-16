#include "LBMSolver.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include <queue>

#include "LBMKernels.cuh"
#include "colorMap.cuh"

#include <cmath>
#include <fstream>
#include <thrust/device_vector.h>


#define THREADS_PER_BLOCK 8

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

LBMSolver::LBMSolver( uint nx, uint ny, float omega, float U, float V )
    : nx(nx), ny(ny), omega(omega), U(U), V(V), lbModel('c'), geoMode('w')
{
	this->speed = sqrtf(U * U + V * V);
	this->alpha = acosf(U / sqrt((U*U + V * V + 1.e-20)));
    this->nx = nx;
    this->ny = ny;
    this->refLength=ny;

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

    this->minVelocity =  0.0f;
    this->maxVelocity =  0.02f;

    checkCudaErrors( cudaMemcpyToSymbol( colorMapDeviceR, colorMapHostR, 36*sizeof(float) ) );
    checkCudaErrors( cudaMemcpyToSymbol( colorMapDeviceG, colorMapHostG, 36*sizeof(float) ) );
    checkCudaErrors( cudaMemcpyToSymbol( colorMapDeviceB, colorMapHostB, 36*sizeof(float) ) );
}

LBMSolver::~LBMSolver()
{
}
void LBMSolver::connectVertexBuffer(uint vertexBufferID)
{
    cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&this->glVertexBufferResource, 
                                                          vertexBufferID, 
                                                          cudaGraphicsMapFlagsNone);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(cudaStatus));
        // Additional error handling if necessary
    }
}

// void LBMSolver::connectVertexBuffer(uint vertexBufferID)
// {
//     // register the OpenGL vertex Buffer within CUDA
//     checkCudaErrors( cudaGraphicsGLRegisterBuffer(&this->glVertexBufferResource, 
//                                                   vertexBufferID, 
//                                                   cudaGraphicsMapFlagsNone) );
    
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LBMSolver::initializeDistributions()
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

void LBMSolver::initializeGeo()
{
    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    initializeGeoKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny );

    scaleColorMap();
}

void LBMSolver::collision()
{
    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    //////////////////////////////////////////////////////////////////////////

    collisionKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny, this->omega, this->U, this->V, this->lbModel );
    getLastCudaError("collisionKernel failed.");

    //////////////////////////////////////////////////////////////////////////

    swap( f.f0n, f.f0p );
    swap( f.fnn, f.fpp );
    swap( f.fp0, f.fn0 );
    swap( f.fpn, f.fnp );
}

void LBMSolver::postProcessing( char type )
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

    postProcessingSetColorKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny, verticesDev, type, min, max, this->geoMode );
    getLastCudaError("postProcessingSetColorKernel failed.");

    cudaGraphicsUnmapResources(1, &this->glVertexBufferResource, 0);
    getLastCudaError("cudaGraphicsUnmapResources failed");
}

void LBMSolver::computeMacroscopicQuantities()
{
    dim3 threads( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
    dim3 blocks ( ( this->nx +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK,
                  ( this->ny +  THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK );

    postProcessingMacroscopicQuantitiesKernel<<<blocks, threads>>>( this->getDistPtr(), this->nx, this->ny );
    getLastCudaError("postProcessingMacroscopicQuantitiesKernel failed.");
}

void LBMSolver::scaleColorMap()
{
    this->computeMacroscopicQuantities();

    this->minPressure = thrust::reduce( this->f.pressure->begin(), this->f.pressure->end(),  10.0f, thrust::minimum<float>() );
    this->maxPressure = thrust::reduce( this->f.pressure->begin(), this->f.pressure->end(), -10.0f, thrust::maximum<float>() );

    std::cout << "Pressure = ( " << this->minPressure << ", " << this->maxPressure << " )" << std::endl;

    this->minVelocity = thrust::reduce( this->f.velocity->begin(), this->f.velocity->end(),  10.0f, thrust::minimum<float>() );
    this->maxVelocity = thrust::reduce( this->f.velocity->begin(), this->f.velocity->end(), -10.0f, thrust::maximum<float>() );

    std::cout << "Velocity = ( " << this->minVelocity << ", " << this->maxVelocity << " )" << std::endl;
}

void LBMSolver::setGeo(uint xIdx, uint yIdx, char geo)
{
    dim3 threads ( 2, 2 );

    if( geo == GEO_FLUID ){
        threads.x += 4;
        threads.y += 4;
    }

    setGeoKernel<<<1, threads>>>( this->getDistPtr(), this->nx, this->ny, xIdx, yIdx, geo );
}

void LBMSolver::setGeo(uint xIdx1, uint yIdx1, uint xIdx2, uint yIdx2, char geo)
{
    int dxIdx = xIdx2 - xIdx1;
    int dyIdx = yIdx2 - yIdx1;

    if( abs(dxIdx) >= abs(dyIdx) ){
        for( uint idx = 0; idx < abs(dxIdx); idx++ ){
    
            float xInc = ( dxIdx != 0 )?( float(dxIdx) / float( abs(dxIdx) ) ):(0);
            float yInc = ( dxIdx != 0 )?( float(dyIdx) / float( abs(dxIdx) ) ):(0);
            
            // int x = int(xIdx1) + float(idx) * xInc;
            // int y = int(yIdx1) + float(idx) * yInc;
            float x = xIdx1 + static_cast<float>(idx) * xInc;
            float y = yIdx1 + static_cast<float>(idx) * yInc;

            this->setGeo(x,y, geo);
        }
    }else{
        for( uint idx = 0; idx < abs(dyIdx); idx++ ){
    
            float xInc = ( dyIdx != 0 )?( float(dxIdx) / float( abs(dyIdx) ) ):(0);
            float yInc = ( dyIdx != 0 )?( float(dyIdx) / float( abs(dyIdx) ) ):(0);

            // int x = int(xIdx1) + float(idx) * xInc;
            // int y = int(yIdx1) + float(idx) * yInc;
            float x = xIdx1 + static_cast<float>(idx) * xInc;
            float y = yIdx1 + static_cast<float>(idx) * yInc;

            this->setGeo(x,y, geo);
        }
    }
}

void LBMSolver::setGeoFloodFill(uint xIdx, uint yIdx, char geo)
{
    // based on
    // https://stackoverflow.com/questions/30608448/flood-fill-recursive-stack-overflow

    // download geo field
    charVecHost hostGeo = *this->f.geo;

    //setGeoFloodFillRecursion( xIdx, yIdx, geo, hostGeo );

    struct coordinate { uint x, y; };
    std::queue<coordinate> to_draw;
    to_draw.push({xIdx, yIdx});

    while (!to_draw.empty())
    {
        auto top = to_draw.front();
        to_draw.pop();

        if( top.x <= 0 || top.y <= 0 || top.x >= nx - 2 || top.y >= ny - 2 ) continue;

        uint nodeIdx = top.x + top.y * nx;

        if( hostGeo[ nodeIdx ] == geo ) continue;

        hostGeo[ nodeIdx ] = geo;

        to_draw.push( { top.x, top.y + 1 } );
        to_draw.push( { top.x, top.y - 1 } );
        to_draw.push( { top.x + 1, top.y } );
        to_draw.push( { top.x - 1, top.y } );
    }

    // upload geo field
    *this->f.geo = hostGeo;
}


void LBMSolver::setNu(float nu)
{
    if( nu < 1.0e-8f ) nu = 1.0e-8f;
    if( nu > 0.1f )     nu = 0.1f;
    this->omega = 1.0f / ( 3.0f * nu + 0.5f );
}

float LBMSolver::getNu()
{
    return ( 1.0f / this->omega - 0.5f ) / 3.0f;
}

void LBMSolver::setU(float U)
{
    if( U >  0.1f ) U =  0.1f;
    if( U < -0.1f ) U = -0.1f;
    this->U = U;
}

void LBMSolver::setV(float V)
{
    if( V >  0.1f ) V =  0.1f;
    if( V < -0.1f ) V = -0.1f;
    this->V = V;
}

void LBMSolver::setAlpha(float alpha)
{
	this->alpha = alpha;
}

void LBMSolver::setSpeed(float speed)
{
	this->speed = speed;
	setU(speed*cosf(alpha));
	setV(speed*sinf(alpha));
}

void LBMSolver::setRefLength(uint ref)
{
	this->refLength = ref;
}

float LBMSolver::getU()
{
    return this->U;
}

float LBMSolver::getV()
{
    return this->V;
}

float LBMSolver::getAlpha()
{
	return this->alpha;
}

float LBMSolver::getSpeed()
{
	return this->speed;
}

uint LBMSolver::getRefLength()
{
	return this->refLength;
}

void LBMSolver::setLBModel(char lbModel)
{
    this->lbModel = lbModel;
}

char LBMSolver::getLBModel()
{
    return this->lbModel;
}

void LBMSolver::setGeoMode(char geoMode)
{
    this->geoMode = geoMode;
}

char LBMSolver::getGeoMode()
{
    return this->geoMode;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LBMSolver::swap( floatVecPtr& lhs, floatVecPtr& rhs )
{
    floatVecPtr tmp = lhs;
    lhs = rhs;
    rhs = tmp;
}

uint LBMSolver::c2i( uint xIdx, uint yIdx )
{
    return yIdx * this->nx + xIdx;
}

D2Q9Ptr LBMSolver::getDistPtr()
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

float LBMSolver::getVelocity(uint index)
{
    thrust::device_vector<float> velocity = *f.velocity;
    return velocity[index];
}

float LBMSolver::getPressure(uint index)
{
    thrust::device_vector<float> pressure = *f.pressure;
    return pressure[index];
}
 std::vector<float> LBMSolver::getVelocityData() { 
    thrust::device_vector<float>& velocityData = *f.velocity;
    std::vector<float> velocityDataHost(velocityData.begin(), velocityData.end());
    return velocityDataHost;
    
    
     }
     std::vector<float> LBMSolver::getPressureData() { 
    thrust::device_vector<float>& PressureData = *f.pressure;
    std::vector<float> PressureDataHost(PressureData.begin(), PressureData.end());
    return PressureDataHost;
    
    
     }
   

   std::vector<char> LBMSolver::getgeoData() {
        thrust::device_vector<char>& geoprofile = *f.geo;
        std::vector<char> geoprofileHost(geoprofile.size());
        thrust::copy(geoprofile.begin(), geoprofile.end(), geoprofileHost.begin());

        return geoprofileHost;
    }

void LBMSolver::setgeoData(const std::vector<char>& data) {
        thrust::device_vector<char>& geoprofile = *f.geo;
        geoprofile.resize(data.size());
        thrust::copy(data.begin(), data.end(), geoprofile.begin());
    }

 


