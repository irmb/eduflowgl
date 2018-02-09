#include "lbm.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initializeDistributionsKernel( D2Q9Ptr f, uint nx, uint ny, float U, float V )
{
    uint xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( xIdx >= nx - 1 || yIdx >= ny - 1 ) return;

    uint nodeIdx = yIdx * nx + xIdx;
    
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

    if      ( xIdx == nx - 1 || yIdx == ny - 1 )                                     f.geo[ nodeIdx ] = 1;
    else if ( yIdx == 0      || xIdx == 0      || xIdx == nx - 2 || yIdx == ny - 2 ) f.geo[ nodeIdx ] = 2;
    else                                                                             f.geo[ nodeIdx ] = 0;
}

__global__ void collisionKernel( D2Q9Ptr f, uint nx, uint ny, float omega, float U0, float V0 )
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

    if( geo == 1 ) return;

    //////////////////////////////////////////////////////////////////////////

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
        U = U0;
        V = V0;

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
        U = 0.0f;

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
}

__global__ void postProcessingMacroscopicQuantitiesKernel( D2Q9Ptr f, uint nx, uint ny )
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

    if( geo != 0 ){
        f.pressure[ nodeIdx00 ] = 0.0;
        f.velocity[ nodeIdx00 ] = 0.0;

        return;
    }

    //////////////////////////////////////////////////////////////////////////

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

    f.pressure[ nodeIdx00 ] = dRho;
    f.velocity[ nodeIdx00 ] = sqrt( U * U + V * V );
}

__global__ void postProcessingSetColorKernel( D2Q9Ptr f, uint nx, uint ny, float* vertices, char type, float min, float max )
{
    uint xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( xIdx >= nx - 1 || yIdx >= ny - 1 ) return;
    
    uint nodeIdx00 = ( xIdx     ) + ( yIdx     ) * nx;

    //////////////////////////////////////////////////////////////////////////

    char  geo = f.geo[ nodeIdx00 ];

    if( geo == 1 ){
        vertices[ 5 * nodeIdx00 + 2 ] = 0.0f;
        vertices[ 5 * nodeIdx00 + 4 ] = 0.0f;
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    
    float value;

    if( type == 'p' ) value = f.pressure[ nodeIdx00 ];
    else              value = f.velocity[ nodeIdx00 ];

    value = ( value - min ) / ( max - min );

    vertices[ 5 * nodeIdx00 + 2 ] =        value;
    vertices[ 5 * nodeIdx00 + 4 ] = 1.0f - value;
}

__global__ void setGeoKernel( D2Q9Ptr f, uint nx, uint ny, uint x, uint y, char geo )
{
    int xIdx =  threadIdx.x - 0.5 * blockDim.x;
    int yIdx =  threadIdx.y - 0.5 * blockDim.y;

    uint r = sqrt( float(xIdx * xIdx + yIdx * yIdx) );

    if( r > 0.5 * blockDim.x ) return;

    xIdx += x;
    yIdx += y;

    if( xIdx < 0 || yIdx < 0 || xIdx >= nx - 1 || yIdx >= ny - 1 ) return;

    uint nodeIdx = ( xIdx     ) + ( yIdx     ) * nx;

    f.geo[ nodeIdx ] = geo;
}
