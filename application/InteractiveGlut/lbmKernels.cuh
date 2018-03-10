#ifndef lbmKernels_H
#define lbmKernels_H

#include "lbm.h"

#if defined(__APPLE__) || defined(MACOSX)
#include <GL/glew.h>
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include "colorMap.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline void readDistributionEsoTwist( const D2Q9Ptr& f, 
                                                 D2Q9Distribution& g, 
                                                 const uint& nodeIdx00, 
                                                 const uint& nodeIdx0p, 
                                                 const uint& nodeIdxp0, 
                                                 const uint& nodeIdxpp )
{
    g.f00 = f.f00[ nodeIdx00 ];
    g.f0p = f.f0p[ nodeIdx00 ];
    g.fpp = f.fpp[ nodeIdx00 ];
    g.fp0 = f.fp0[ nodeIdx00 ];

    g.fn0 = f.fn0[ nodeIdxp0 ];
    g.fnp = f.fnp[ nodeIdxp0 ];

    g.fpn = f.fpn[ nodeIdx0p ];
    g.f0n = f.f0n[ nodeIdx0p ];

    g.fnn = f.fnn[ nodeIdxpp ];
}

__device__ inline void writeDistributionEsoTwist( const D2Q9Ptr& f, 
                                                  D2Q9Distribution& g, 
                                                  const uint& nodeIdx00, 
                                                  const uint& nodeIdx0p, 
                                                  const uint& nodeIdxp0, 
                                                  const uint& nodeIdxpp )
{
    f.f00[ nodeIdx00 ] = g.f00;
    f.f0p[ nodeIdx00 ] = g.f0n;
    f.fpp[ nodeIdx00 ] = g.fnn;
    f.fp0[ nodeIdx00 ] = g.fn0;

    f.fn0[ nodeIdxp0 ] = g.fp0;
    f.fnp[ nodeIdxp0 ] = g.fpn;

    f.fpn[ nodeIdx0p ] = g.fnp;
    f.f0n[ nodeIdx0p ] = g.f0p;

    f.fnn[ nodeIdxpp ] = g.fpp;
}

__device__ inline void writeDistributionSelf( const D2Q9Ptr& f, 
                                              D2Q9Distribution& g, 
                                              const uint nodeIdx )
{
    f.f00[ nodeIdx ] = g.f00;
    f.fp0[ nodeIdx ] = g.fp0;
    f.fn0[ nodeIdx ] = g.fn0;
    f.f0p[ nodeIdx ] = g.f0p;
    f.f0n[ nodeIdx ] = g.f0n;
    f.fpp[ nodeIdx ] = g.fpp;
    f.fpn[ nodeIdx ] = g.fpn;
    f.fnn[ nodeIdx ] = g.fnn;
    f.fnp[ nodeIdx ] = g.fnp;
}

__device__ inline void setEquilibrium( D2Q9Distribution& g,
                                       const float& U,
                                       const float& V )
{
    g.f00 = (   (-2       + 3*(U*U) ) * (-2       + 3*(V*V) ) ) / 9.0  - 4.0f/9.0f  ;
    g.fp0 = ( - ( 1 + 3*U + 3*(U*U) ) * (-2       + 3*(V*V) ) ) / 18.0 - 1.0f/9.0f  ;
    g.fn0 = ( - ( 1 - 3*U + 3*(U*U) ) * (-2       + 3*(V*V) ) ) / 18.0 - 1.0f/9.0f  ;
    g.f0p = ( - (-2       + 3*(U*U) ) * ( 1 + 3*V + 3*(V*V) ) ) / 18.0 - 1.0f/9.0f  ;
    g.f0n = ( - (-2       + 3*(U*U) ) * ( 1 - 3*V + 3*(V*V) ) ) / 18.0 - 1.0f/9.0f  ;
    g.fpp = (   ( 1 + 3*U + 3*(U*U) ) * ( 1 + 3*V + 3*(V*V) ) ) / 36.0 - 1.0f/36.0f ;
    g.fpn = (   ( 1 + 3*U + 3*(U*U) ) * ( 1 - 3*V + 3*(V*V) ) ) / 36.0 - 1.0f/36.0f ;
    g.fnn = (   ( 1 - 3*U + 3*(U*U) ) * ( 1 - 3*V + 3*(V*V) ) ) / 36.0 - 1.0f/36.0f ;
    g.fnp = (   ( 1 - 3*U + 3*(U*U) ) * ( 1 + 3*V + 3*(V*V) ) ) / 36.0 - 1.0f/36.0f ;
}

__device__ inline void computeMacroscopicQuantities( const D2Q9Distribution& g, 
                                                     float& dRho, 
                                                     float& U, 
                                                     float& V )
{
    dRho = ( ( (   g.fnp + g.fpn ) + (   g.fnn + g.fpp ) ) + ( ( g.f0p + g.f0n ) + ( g.fp0 + g.fn0 ) ) ) + g.f00;
    U    = ( ( ( - g.fnp + g.fpn ) + ( - g.fnn + g.fpp ) ) + (                     ( g.fp0 - g.fn0 ) ) ) / ( 1.0f + dRho );
    V    = ( ( (   g.fnp - g.fpn ) + ( - g.fnn + g.fpp ) ) + ( ( g.f0p - g.f0n )                     ) ) / ( 1.0f + dRho );
}

__device__ inline void forwardCentralMomentTransform( D2Q9Distribution& g,
                                                      const float& U,
                                                      const float& V )
{
    float n1,n2;
    n1    =         (g.fnp+g.fnn) + g.fn0;
    n2    =         (g.fnp-g.fnn)         -   V*(n1+1.0f/6.0f);
    g.fnp =         (g.fnp+g.fnn)
            -2.0f*V*(g.fnp-g.fnn)         + V*V*(n1+1.0f/6.0f);
    g.fnn = n1;
    g.fn0 = n2;

    n1    =         (g.f0p+g.f0n) + g.f00;
    n2    =         (g.f0p-g.f0n)         -   V*(n1+2.0f/3.0f);
    g.f0p =         (g.f0p+g.f0n)
            -2.0f*V*(g.f0p-g.f0n)         + V*V*(n1+2.0f/3.0f);
    g.f0n = n1;
    g.f00 = n2;

    n1    =         (g.fpp+g.fpn) + g.fp0;
    n2    =         (g.fpp-g.fpn)         -   V*(n1+1.0f/6.0f);
    g.fpp =         (g.fpp+g.fpn)
            -2.0f*V*(g.fpp-g.fpn)         + V*V*(n1+1.0f/6.0f);
    g.fpn = n1;
    g.fp0 = n2;

    //////////////////////////////////////////////////////////////////////////

    n1    =         (g.fpn+g.fnn) + g.f0n;
    n2    =         (g.fpn-g.fnn)         -   U*(n1+1.0f     );
    g.fpn =         (g.fpn+g.fnn)   
            -2.0f*U*(g.fpn-g.fnn)         + U*U*(n1+1.0f     );
    g.fnn = n1;
    g.f0n = n2;

    n1    =         (g.fp0+g.fn0) + g.f00;
    n2    =         (g.fp0-g.fn0)         -   U*(n1          );
    g.fp0 =         (g.fp0+g.fn0)   
            -2.0f*U*(g.fp0-g.fn0)         + U*U*(n1          );
    g.fn0 = n1;
    g.f00 = n2;

    n1    =         (g.fpp+g.fnp) + g.f0p;
    n2    =         (g.fpp-g.fnp)         -   U*(n1+1.0f/3.0f);
    g.fpp =         (g.fpp+g.fnp)   
            -2.0f*U*(g.fpp-g.fnp)         + U*U*(n1+1.0f/3.0f);
    g.fnp = n1;
    g.f0p = n2;
}

__device__ inline void backwardCentralMomentTransform( D2Q9Distribution& g,
                                                      const float& U,
                                                      const float& V )
{
    float n1, n2;

    n1    =            g.fnn    *(1.0f-U*U) - 2.0f*U*g.f0n               - g.fpn -  U*U;
    n2    = 0.5f * ( ( g.fnn + 1.0f )*(U*U-U) +      g.f0n*(2.0f*U-1.0f) + g.fpn );
    g.fpn = 0.5f * ( ( g.fnn + 1.0f )*(U*U+U) +      g.f0n*(2.0f*U+1.0f) + g.fpn );
    g.f0n = n1;
    g.fnn = n2;

    n1    =            g.fn0    *(1.0f-U*U) - 2.0f*U*g.f00               - g.fp0;
    n2    = 0.5f * ( ( g.fn0        )*(U*U-U) +      g.f00*(2.0f*U-1.0f) + g.fp0 );
    g.fp0 = 0.5f * ( ( g.fn0        )*(U*U+U) +      g.f00*(2.0f*U+1.0f) + g.fp0 );
    g.f00 = n1;
    g.fn0 = n2;

    n1    =            g.fnp        *(1.0f-U*U) - 2.0f*U*g.f0p               - g.fpp -  U*U*1.0f/3.0f;
    n2    = 0.5f * ( ( g.fnp + 1.0f/3.0 )*(U*U-U) +      g.f0p*(2.0f*U-1.0f) + g.fpp );
    g.fpp = 0.5f * ( ( g.fnp + 1.0f/3.0 )*(U*U+U) +      g.f0p*(2.0f*U+1.0f) + g.fpp );
    g.f0p = n1;
    g.fnp = n2;

    //////////////////////////////////////////////////////////////////////////

    n1    =            g.fnn         *(1.0f-V*V) - 2.0f*V*g.fn0               - g.fnp -  V*V*1.0f/6.0f;
    n2    = 0.5f * ( ( g.fnn + 1.0f/6.0f )*(V*V-V) +      g.fn0*(2.0f*V-1.0f) + g.fnp );
    g.fnp = 0.5f * ( ( g.fnn + 1.0f/6.0f )*(V*V+V) +      g.fn0*(2.0f*V+1.0f) + g.fnp );
    g.fn0 = n1;
    g.fnn = n2;

    n1    =            g.f0n         *(1.0f-V*V) - 2.0f*V*g.f00               - g.f0p -  V*V*2.0f/3.0f;
    n2    = 0.5f * ( ( g.f0n + 2.0f/3.0f )*(V*V-V) +      g.f00*(2.0f*V-1.0f) + g.f0p );
    g.f0p = 0.5f * ( ( g.f0n + 2.0f/3.0f )*(V*V+V) +      g.f00*(2.0f*V+1.0f) + g.f0p );
    g.f00 = n1;
    g.f0n = n2;

    n1    =            g.fpn         *(1.0f-V*V) - 2.0f*V*g.fp0               - g.fpp -  V*V*1.0f/6.0f;
    n2    = 0.5f * ( ( g.fpn + 1.0f/6.0f )*(V*V-V) +      g.fp0*(2.0f*V-1.0f) + g.fpp );
    g.fpp = 0.5f * ( ( g.fpn + 1.0f/6.0f )*(V*V+V) +      g.fp0*(2.0f*V+1.0f) + g.fpp );
    g.fp0 = n1;
    g.fpn = n2;
}

__device__ inline void collisonCentralMoments( D2Q9Distribution& g,
                                               float U,
                                               float V,
                                               float omega )
{
    forwardCentralMomentTransform( g, U, V );

    //////////////////////////////////////////////////////////////////////////

    float dxU=-omega*0.5f*(2.0f*g.fpn-g.fnp)-0.5f*(g.fpn+g.fnp-2.0f/3.0f*g.fnn);
    float dyV=-omega*0.5f*(2.0f*g.fnp-g.fpn)-0.5f*(g.fpn+g.fnp-2.0f/3.0f*g.fnn);

    float mXXMYY=g.fpn-g.fnp;
    float mXXPYY=2.0f/3.0f*g.fnn-3.0f/2.0f*(U*U*dxU+V*V*dyV);//g.fpn+g.fnp;

    mXXMYY=(1.0f-omega)*mXXMYY-3.0f*(1.0f-omega*0.5f)*(U*U*dxU+V*V*dyV);

    g.fpn=( mXXMYY+mXXPYY)*0.5f;
    g.fnp=(-mXXMYY+mXXPYY)*0.5f;

    //g.fpn = 1.0f/3.0f * g.fnn * omega + ( 1.0f - omega ) * g.fpn;

    //g.fnp = 1.0f/3.0f * g.fnn * omega + ( 1.0f - omega ) * g.fnp;

    g.f00 =                           ( 1.0f - omega ) * g.f00;

    g.fpp = 1.0f/9.0f * g.fnn;

    /////////////////////////////////////////////////////////////////////////
    float omega3 = 8.0f * ( omega - 2.0f ) / ( omega - 8.0f );
    omega3 = omega3 +(1.0f-omega3)*fabs(g.f0p)/(fabs(g.f0p)+0.0001f);

    g.f0p *= ( 1.0f - omega3 );

    omega3 = 8.0f * ( omega - 2.0f ) / ( omega - 8.0f );
    omega3 = omega3 +(1.0f-omega3)*fabs(g.fp0)/(fabs(g.fp0)+0.0001f);
    g.fp0 *= ( 1.0f - omega3 );
    /////////////////////////////////////////////////////////////////////

    //g.f0p = 0.0f;

    //g.fp0 = 0.0f;

    //////////////////////////////////////////////////////////////////////////

    backwardCentralMomentTransform( g, U, V );

    //////////////////////////////////////////////////////////////////////////
}

__device__ inline void collisionBGK( D2Q9Distribution& g,
                                     float U,
                                     float V,
                                     float dRho,
                                     float omega )
{
    g.f00 = g.f00 * ( 1.0f - omega ) + omega * (   ( ( 1.0f + dRho ) * (-2.0f          + 3.0f*(U*U) )*(-2.0f          + 3.0f*(V*V) ) ) / 9.0f  - 4.0f/9.0f  );
    g.fp0 = g.fp0 * ( 1.0f - omega ) + omega * ( - ( ( 1.0f + dRho ) * ( 1.0f + 3.0f*U + 3.0f*(U*U) )*(-2.0f          + 3.0f*(V*V) ) ) / 18.0f - 1.0f/9.0f  );
    g.fn0 = g.fn0 * ( 1.0f - omega ) + omega * ( - ( ( 1.0f + dRho ) * ( 1.0f - 3.0f*U + 3.0f*(U*U) )*(-2.0f          + 3.0f*(V*V) ) ) / 18.0f - 1.0f/9.0f  );
    g.f0p = g.f0p * ( 1.0f - omega ) + omega * ( - ( ( 1.0f + dRho ) * (-2.0f          + 3.0f*(U*U) )*( 1.0f + 3.0f*V + 3.0f*(V*V) ) ) / 18.0f - 1.0f/9.0f  );
    g.f0n = g.f0n * ( 1.0f - omega ) + omega * ( - ( ( 1.0f + dRho ) * (-2.0f          + 3.0f*(U*U) )*( 1.0f - 3.0f*V + 3.0f*(V*V) ) ) / 18.0f - 1.0f/9.0f  );
    g.fpp = g.fpp * ( 1.0f - omega ) + omega * (   ( ( 1.0f + dRho ) * ( 1.0f + 3.0f*U + 3.0f*(U*U) )*( 1.0f + 3.0f*V + 3.0f*(V*V) ) ) / 36.0f - 1.0f/36.0f );
    g.fpn = g.fpn * ( 1.0f - omega ) + omega * (   ( ( 1.0f + dRho ) * ( 1.0f + 3.0f*U + 3.0f*(U*U) )*( 1.0f - 3.0f*V + 3.0f*(V*V) ) ) / 36.0f - 1.0f/36.0f );
    g.fnn = g.fnn * ( 1.0f - omega ) + omega * (   ( ( 1.0f + dRho ) * ( 1.0f - 3.0f*U + 3.0f*(U*U) )*( 1.0f - 3.0f*V + 3.0f*(V*V) ) ) / 36.0f - 1.0f/36.0f );
    g.fnp = g.fnp * ( 1.0f - omega ) + omega * (   ( ( 1.0f + dRho ) * ( 1.0f - 3.0f*U + 3.0f*(U*U) )*( 1.0f + 3.0f*V + 3.0f*(V*V) ) ) / 36.0f - 1.0f/36.0f );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initializeDistributionsKernel( D2Q9Ptr f, uint nx, uint ny, float U, float V )
{
    uint xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( xIdx >= nx - 1 || yIdx >= ny - 1 ) return;

    uint nodeIdx = yIdx * nx + xIdx;
    
    D2Q9Distribution g;

    setEquilibrium(g, U, V);

    writeDistributionSelf(f, g, nodeIdx);
}

__global__ void initializeGeoKernel( D2Q9Ptr f, uint nx, uint ny )
{
    uint xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( xIdx >= nx - 1 || yIdx >= ny - 1 ) return;

    uint nodeIdx = yIdx * nx + xIdx;

    //if      ( xIdx == nx - 1 || yIdx == ny - 1 ) f.geo[ nodeIdx ] = 1;
    //else if ( xIdx == 0      || xIdx == nx - 2 ) f.geo[ nodeIdx ] = 2;
    //else if ( yIdx == 0      || yIdx == ny - 2 ) f.geo[ nodeIdx ] = 3;
    //else                                         f.geo[ nodeIdx ] = 0;

    if      ( xIdx == nx - 1 || yIdx == ny - 1 )                                     f.geo[ nodeIdx ] = 1;
    else if ( yIdx == 0      || xIdx == 0      || xIdx == nx - 2 || yIdx == ny - 2 ) f.geo[ nodeIdx ] = 2;
    else                                                                             f.geo[ nodeIdx ] = 0;
}

__global__ void collisionKernel( D2Q9Ptr f, uint nx, uint ny, float omega, float U, float V, char model )
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

    D2Q9Distribution g;

    readDistributionEsoTwist( f, g, nodeIdx00, nodeIdx0p, nodeIdxp0, nodeIdxpp );

    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////

    if( geo == 0 ){

        float dRho;

        computeMacroscopicQuantities( g, dRho, U, V );

        // Sponge Layer
        if( nx - xIdx < 50 ) omega = 1.0f - float( nx - xIdx )/50.0f * ( 1.0f - omega );

        if( model == 'c' ) collisonCentralMoments( g, U, V, omega );
        else               collisionBGK(g, U, V, dRho, omega);
    }
    else if( geo == 2 ){
        setEquilibrium(g, U, V);
    }
    else if( geo == 3 ){
        setEquilibrium(g, 0.0f, 0.0f);
    }

    //////////////////////////////////////////////////////////////////////////

    writeDistributionEsoTwist( f, g, nodeIdx00, nodeIdx0p, nodeIdxp0, nodeIdxpp );
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
        f.pressure[ nodeIdx00 ] = 0.0f;
        f.velocity[ nodeIdx00 ] = 0.0f;

        return;
    }

    //////////////////////////////////////////////////////////////////////////

    D2Q9Distribution g;

    readDistributionEsoTwist( f, g, nodeIdx00, nodeIdx0p, nodeIdxp0, nodeIdxpp );

    float dRho, U, V;

    computeMacroscopicQuantities(g, dRho, U, V);

    //////////////////////////////////////////////////////////////////////////

    f.pressure[ nodeIdx00 ] = dRho;
    f.velocity[ nodeIdx00 ] = sqrt( U * U + V * V );
}

__global__ void postProcessingSetColorKernel( D2Q9Ptr f, uint nx, uint ny, float* vertices, char type, float min, float max, char geoMode )
{
    uint xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( xIdx >= nx - 1 || yIdx >= ny - 1 ) return;
    
    uint nodeIdx00 = ( xIdx     ) + ( yIdx     ) * nx;

    //////////////////////////////////////////////////////////////////////////

    char  geo = f.geo[ nodeIdx00 ];

    if( geo == 1 ){
        vertices[ 5 * nodeIdx00 + 2 ] = ( geoMode == 'b' ) ? 0.0f : 1.0f;
        vertices[ 5 * nodeIdx00 + 3 ] = ( geoMode == 'b' ) ? 0.0f : 1.0f;
        vertices[ 5 * nodeIdx00 + 4 ] = ( geoMode == 'b' ) ? 0.0f : 1.0f;
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    
    float value;

    if( type == 'p' ) value = f.pressure[ nodeIdx00 ];
    else              value = f.velocity[ nodeIdx00 ];
    
    if( value < min )
        value = 0.0f;
    else if ( value > max )
        value = 1.0f;
    else
        value = ( value - min ) / ( max - min );

    unsigned int idx           = value * 34;
    float        interpolation = value * 34.0f - float( idx );
    
    float r = ( ( 1.0f - interpolation ) * colorMapDeviceR[idx  ]
              +          interpolation   * colorMapDeviceR[idx+1] ) * 200.0f * 1.2f;

    float g = ( ( 1.0f - interpolation ) * colorMapDeviceG[idx  ]
              +          interpolation   * colorMapDeviceG[idx+1] ) * 197.0f * 1.2f;

    float b = ( ( 1.0f - interpolation ) * colorMapDeviceB[idx  ]
              +          interpolation   * colorMapDeviceB[idx+1] ) * 189.0f * 1.2f;

    vertices[ 5 * nodeIdx00 + 2 ] = r/256.0f;
    vertices[ 5 * nodeIdx00 + 3 ] = g/256.0f;
    vertices[ 5 * nodeIdx00 + 4 ] = b/256.0f;
}

__global__ void setGeoKernel( D2Q9Ptr f, uint nx, uint ny, uint x, uint y, char geo )
{
    int xIdx =  threadIdx.x - blockDim.x / 2;
    int yIdx =  threadIdx.y - blockDim.y / 2;

    uint r = sqrt( float(xIdx * xIdx + yIdx * yIdx) );

    if( r > blockDim.x / 2 ) return;

    xIdx += x;
    yIdx += y;

    if( xIdx < 0 || yIdx < 0 || xIdx >= nx - 1 || yIdx >= ny - 1 ) return;

    uint nodeIdx = ( xIdx     ) + ( yIdx     ) * nx;

    f.geo[ nodeIdx ] = geo;
}

#endif