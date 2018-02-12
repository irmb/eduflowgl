#ifndef lbmKernels_H
#define lbmKernels_H

#include "lbm.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

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

__device__ inline void collisonCascade( float& g00, 
                                        float& gp0, 
                                        float& gn0, 
                                        float& g0p, 
                                        float& g0n, 
                                        float& gpp, 
                                        float& gpn, 
                                        float& gnn, 
                                        float& gnp,
                                        float U,
                                        float V,
                                        float omega )
{
    // parametrization Paper part I, Eq. 6 - 11
    // in central moments: n -> 0, 0 -> 1, p -> 2

    // forwards fast central moment transform

    float n1,n2;
    n1=(gnp+gnn)+gn0;
    n2=(gnp-gnn)-V*(n1+1.0f/6.0f);
    gnp=(gnp+gnn)-2.0f*V*(gnp-gnn)+V*V*(n1+1.0f/6.0f);
    gnn=n1;
    gn0=n2;

    n1=(g0p+g0n)+g00;
    n2=(g0p-g0n)-V*(n1+2.0f/3.0f);
    g0p=(g0p+g0n)-2.0f*V*(g0p-g0n)+V*V*(n1+2.0f/3.0f);
    g0n=n1;
    g00=n2;

    n1=(gpp+gpn)+gp0;
    n2=(gpp-gpn)-V*(n1+1.0f/6.0f);
    gpp=(gpp+gpn)-2.0f*V*(gpp-gpn)+V*V*(n1+1.0f/6.0f);
    gpn=n1;
    gp0=n2;

    n1 =         (gpn+gnn) + g0n;
    n2 =         (gpn-gnn) -   U*(n1+1.0f);
    gpn=         (gpn+gnn)   
         -2.0f*U*(gpn-gnn) + U*U*(n1+1.0f);
    gnn= n1;
    g0n= n2;

    n1 =         (gp0+gn0) + g00;
    n2 =         (gp0-gn0) -   U*(n1);
    gp0=         (gp0+gn0)   
         -2.0f*U*(gp0-gn0) + U*U*(n1);
    gn0= n1;
    g00= n2;

    n1 =         (gpp+gnp) + g0p;
    n2 =         (gpp-gnp) -   U*(n1+1.0f/3.0);
    gpp=         (gpp+gnp)   
         -2.0f*U*(gpp-gnp) + U*U*(n1+1.0f/3.0);
    gnp= n1;
    g0p= n2;

    //////////////////////////////////////////////////////////////////////////

    float dxU=-omega*0.5f*(2.0f*gpn-gnp)-0.5f*(gpn+gnp-2.0f/3.0f*gnn);
    float dyV=-omega*0.5f*(2.0f*gnp-gpn)-0.5f*(gpn+gnp-2.0f/3.0f*gnn);

    float mXXMYY=gpn-gnp;
    float mXXPYY=2.0f/3.0f*gnn-3.0f/2.0f*(U*U*dxU+V*V*dyV);//gpn+gnp;

    mXXMYY=(1.0f-omega)*mXXMYY-3.0f*(1.0f-omega*0.5f)*(U*U*dxU+V*V*dyV);

    gpn=( mXXMYY+mXXPYY)*0.5f;
    gnp=(-mXXMYY+mXXPYY)*0.5f;

    //gpn = 1.0f/3.0f * gnn * omega + ( 1.0f - omega ) * gpn;

    //gnp = 1.0f/3.0f * gnn * omega + ( 1.0f - omega ) * gnp;

    g00 =                           ( 1.0f - omega ) * g00;

    gpp = 1.0f/9.0f * gnn;

    /////////////////////////////////////////////////////////////////////////
    float omega3 = 8.0f * ( omega - 2.0f ) / ( omega - 8.0f );
    omega3 = omega3 +(1.0f-omega3)*fabs(g0p)/(fabs(g0p)+0.0001f);

    g0p *= ( 1.0f - omega3 );

    omega3 = 8.0f * ( omega - 2.0f ) / ( omega - 8.0f );
    omega3 = omega3 +(1.0f-omega3)*fabs(gp0)/(fabs(gp0)+0.0001f);
    gp0 *= ( 1.0f - omega3 );
    /////////////////////////////////////////////////////////////////////

    //g0p = 0.0f;

    //gp0 = 0.0f;

    //////////////////////////////////////////////////////////////////////////

    // backward fast central moment transform

    n1  =            gnn    *(1.0f-U*U) - 2.0f*U*g0n               - gpn -  U*U*1.0f;
    n2  = 0.5f * ( ( gnn + 1.0f )*(U*U-U) +      g0n*(2.0f*U-1.0f) + gpn );
    gpn = 0.5f * ( ( gnn + 1.0f )*(U*U+U) +      g0n*(2.0f*U+1.0f) + gpn );
    g0n = n1;
    gnn = n2;

    n1  =            gn0    *(1.0f-U*U) - 2.0f*U*g00               - gp0;
    n2  = 0.5f * ( ( gn0        )*(U*U-U) +      g00*(2.0f*U-1.0f) + gp0 );
    gp0 = 0.5f * ( ( gn0        )*(U*U+U) +      g00*(2.0f*U+1.0f) + gp0 );
    g00 = n1;
    gn0 = n2;

    n1  =            gnp        *(1.0f-U*U) - 2.0f*U*g0p               - gpp -  U*U*1.0f/3.0f;
    n2  = 0.5f * ( ( gnp + 1.0f/3.0 )*(U*U-U) +      g0p*(2.0f*U-1.0f) + gpp );
    gpp = 0.5f * ( ( gnp + 1.0f/3.0 )*(U*U+U) +      g0p*(2.0f*U+1.0f) + gpp );
    g0p = n1;
    gnp = n2;

    n1  =            gnn         *(1.0f-V*V) - 2.0f*V*gn0               - gnp -  V*V*1.0f/6.0f;
    n2  = 0.5f * ( ( gnn + 1.0f/6.0f )*(V*V-V) +      gn0*(2.0f*V-1.0f) + gnp );
    gnp = 0.5f * ( ( gnn + 1.0f/6.0f )*(V*V+V) +      gn0*(2.0f*V+1.0f) + gnp );
    gn0 = n1;
    gnn = n2;

    n1  =            g0n         *(1.0f-V*V) - 2.0f*V*g00               - g0p -  V*V*2.0f/3.0f;
    n2  = 0.5f * ( ( g0n + 2.0f/3.0f )*(V*V-V) +      g00*(2.0f*V-1.0f) + g0p );
    g0p = 0.5f * ( ( g0n + 2.0f/3.0f )*(V*V+V) +      g00*(2.0f*V+1.0f) + g0p );
    g00 = n1;
    g0n = n2;

    n1  =            gpn         *(1.0f-V*V) - 2.0f*V*gp0               - gpp -  V*V*1.0f/6.0f;
    n2  = 0.5f * ( ( gpn + 1.0f/6.0f )*(V*V-V) +      gp0*(2.0f*V-1.0f) + gpp );
    gpp = 0.5f * ( ( gpn + 1.0f/6.0f )*(V*V+V) +      gp0*(2.0f*V+1.0f) + gpp );
    gp0 = n1;
    gpn = n2;

    //////////////////////////////////////////////////////////////////////////
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

        if( nx - xIdx < 50 ) omega = 1.0f - float( nx - xIdx )/50.0f * ( 1.0f - omega );

        collisonCascade( g00, 
                         gp0, 
                         gn0, 
                         g0p, 
                         g0n, 
                         gpp, 
                         gpn, 
                         gnn, 
                         gnp,
                         U,
                         V,
                         omega );

        //g00 = g00 * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * (-2       + 3*(U*U))*(-2       + 3*(V*V)))/9.  - 4.0f/9.0f  );
        //gp0 = gp0 * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * ( 1 + 3*U + 3*(U*U))*(-2       + 3*(V*V)))/18. - 1.0f/9.0f  );
        //gn0 = gn0 * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * ( 1 - 3*U + 3*(U*U))*(-2       + 3*(V*V)))/18. - 1.0f/9.0f  );
        //g0p = g0p * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * (-2       + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/18. - 1.0f/9.0f  );
        //g0n = g0n * ( 1.0f - omega ) + omega * ( - ( (1.0f + dRho) * (-2       + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/18. - 1.0f/9.0f  );
        //gpp = gpp * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 + 3*U + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
        //gpn = gpn * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 + 3*U + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
        //gnn = gnn * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 - 3*U + 3*(U*U))*( 1 - 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
        //gnp = gnp * ( 1.0f - omega ) + omega * (   ( (1.0f + dRho) * ( 1 - 3*U + 3*(U*U))*( 1 + 3*V + 3*(V*V)))/36. - 1.0f/36.0f );
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
        vertices[ 5 * nodeIdx00 + 2 ] = 1.0f;
        vertices[ 5 * nodeIdx00 + 3 ] = 1.0f;
        vertices[ 5 * nodeIdx00 + 4 ] = 1.0f;
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    
    float value;

    if( type == 'p' ) value = f.pressure[ nodeIdx00 ];
    else              value = f.velocity[ nodeIdx00 ];
        
    //// Color map exported from Paraview
    //const float colorMap[36][3] = 
    ///*  0 */  { { 0.000000000000000000f,   0.000000000000000000f,   0.349020000000000000f  },
    ///*  1 */    { 0.039216000000000001f,   0.062744999999999995f,   0.380392000000000010f  },
    ///*  2 */    { 0.062744999999999995f,   0.117647000000000000f,   0.411764999999999990f  },
    ///*  3 */    { 0.090195999999999998f,   0.184314000000000010f,   0.450979999999999990f  },
    ///*  4 */    { 0.125489999999999990f,   0.262745000000000010f,   0.501960999999999990f  },
    ///*  5 */    { 0.160784000000000010f,   0.337255000000000030f,   0.541175999999999990f  },
    ///*  6 */    { 0.200000000000000010f,   0.396077999999999990f,   0.568626999999999990f  },
    ///*  7 */    { 0.239216000000000010f,   0.454901999999999970f,   0.599999999999999980f  },
    ///*  8 */    { 0.286275000000000000f,   0.521568999999999950f,   0.650980000000000000f  },
    ///*  9 */    { 0.337255000000000030f,   0.592157000000000040f,   0.701960999999999950f  },
    ///* 10 */    { 0.388235000000000000f,   0.654901999999999980f,   0.749020000000000020f  },
    ///* 11 */    { 0.466667000000000000f,   0.737254999999999990f,   0.819608000000000000f  },
    ///* 12 */    { 0.572548999999999970f,   0.819608000000000000f,   0.878430999999999960f  },
    ///* 13 */    { 0.654901999999999980f,   0.866666999999999970f,   0.909803999999999950f  },
    ///* 14 */    { 0.752940999999999970f,   0.917646999999999990f,   0.941176000000000010f  },
    ///* 15 */    { 0.823528999999999960f,   0.956863000000000020f,   0.968627000000000020f  },
    ///* 16 */    { 0.988234999999999970f,   0.960783999999999970f,   0.901961000000000010f  },
    ///* 17 */    { 0.941176000000000010f,   0.984314000000000020f,   0.988234999999999970f  },
    ///* 18 */    { 0.988234999999999970f,   0.945097999999999990f,   0.850979999999999960f  },
    ///* 19 */    { 0.980392000000000040f,   0.898039000000000030f,   0.784313999999999960f  },
    ///* 20 */    { 0.968627000000000020f,   0.835293999999999980f,   0.698038999999999970f  },
    ///* 21 */    { 0.949019999999999970f,   0.733333000000000010f,   0.588234999999999950f  },
    ///* 22 */    { 0.929412000000000020f,   0.650980000000000000f,   0.509804000000000030f  },
    ///* 23 */    { 0.909803999999999950f,   0.564706000000000040f,   0.435294000000000010f  },
    ///* 24 */    { 0.878430999999999960f,   0.458824000000000010f,   0.352941000000000000f  },
    ///* 25 */    { 0.839215999999999960f,   0.388235000000000000f,   0.286275000000000000f  },
    ///* 26 */    { 0.760784000000000020f,   0.294117999999999990f,   0.211765000000000010f  },
    ///* 27 */    { 0.701960999999999950f,   0.211765000000000010f,   0.168627000000000000f  },
    ///* 28 */    { 0.650980000000000000f,   0.156863000000000000f,   0.129412000000000000f  },
    ///* 29 */    { 0.599999999999999980f,   0.094117999999999993f,   0.094117999999999993f  },
    ///* 30 */    { 0.549019999999999950f,   0.066667000000000004f,   0.098039000000000001f  },
    ///* 31 */    { 0.501960999999999990f,   0.050979999999999998f,   0.125489999999999990f  },
    ///* 32 */    { 0.450979999999999990f,   0.054901999999999999f,   0.172549000000000010f  },
    ///* 33 */    { 0.400000000000000020f,   0.054901999999999999f,   0.192156999999999990f  },
    ///* 34 */    { 0.349020000000000000f,   0.070587999999999998f,   0.211765000000000010f  },
    //            { 0.349020000000000000f,   0.070587999999999998f,   0.211765000000000010f  } };
    
    if( value < min )
        value = 0.0f;
    else if ( value > max )
        value = 1.0f;
    else
        value = ( value - min ) / ( max - min );

    unsigned int idx           = value * 34;
    float        interpolation = value * 34.0 - float( idx );
    
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

#endif