#ifndef LBM_H
#define LBM_H

#include <memory>

typedef unsigned int uint;

class cudaGraphicsResource;

struct D2Q9
{
    float* f00;
    float* fp0;
    float* fn0;
    float* fpp;
    float* fnp;
    float* fpn;
    float* fnn;
    float* f0p;
    float* f0n;

    char* geo;

    char* geoHost;
};

class lbmSolver
{
private:

    D2Q9 f;

    uint nx;
    uint ny;

    cudaGraphicsResource* glVertexBufferResource; // handles OpenGL-CUDA exchange

public:

    lbmSolver( uint nx, uint ny );
    ~lbmSolver();

    void connectVertexBuffer( uint vertexBufferID );

    void initializeDistributions();

    void collision();

    void postProcessing( char type );

    void swap( float** lhs, float** rhs );

    void setGeo( uint xIdx, uint yIdx, char geo );

    void uploadGeo();
    void downloadGeo();

    uint c2i( uint xIdx, uint yIdx );
};

typedef std::shared_ptr<lbmSolver> lbmSolverPtr;

#endif