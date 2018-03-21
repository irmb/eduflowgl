#ifndef LBM_H
#define LBM_H

#include <memory>
#include <thrust/device_vector.h>

typedef unsigned int uint;

typedef                  thrust::device_vector<float>   floatVec;
typedef std::shared_ptr< thrust::device_vector<float> > floatVecPtr;

typedef                  thrust::device_vector<char>    charVec;
typedef std::shared_ptr< thrust::device_vector<char> >  charVecPtr;

typedef                  thrust::host_vector<char>      charVecHost;

typedef thrust::device_ptr<float> floatPtr;
typedef thrust::device_ptr<char>  charPtr;

class cudaGraphicsResource;

struct D2Q9Memory
{
    floatVecPtr f00;
    floatVecPtr fp0;
    floatVecPtr fn0;
    floatVecPtr fpp;
    floatVecPtr fnp;
    floatVecPtr fpn;
    floatVecPtr fnn;
    floatVecPtr f0p;
    floatVecPtr f0n;

    charVecPtr  geo;

    floatVecPtr pressure;
    floatVecPtr velocity;
};

struct D2Q9Ptr
{
    floatPtr f00;
    floatPtr fp0;
    floatPtr fn0;
    floatPtr fpp;
    floatPtr fnp;
    floatPtr fpn;
    floatPtr fnn;
    floatPtr f0p;
    floatPtr f0n;

    charPtr  geo;

    floatPtr pressure;
    floatPtr velocity;
};

struct D2Q9Distribution
{
    float f00;
    float fp0;
    float fn0;
    float fpp;
    float fnp;
    float fpn;
    float fnn;
    float f0p;
    float f0n;
};

class lbmSolver
{
private:

    D2Q9Memory f;

    uint nx;
    uint ny;
    uint refLength;

    cudaGraphicsResource* glVertexBufferResource; // handles OpenGL-CUDA exchange

    float minVelocity;
    float maxVelocity;

    float minPressure;
    float maxPressure;

    float omega;
    float U;
    float V;
	float alpha;
	float speed;

    char lbModel;
    char geoMode;

public:

    lbmSolver( uint nx, uint ny, float omega, float U, float V );
    ~lbmSolver();

    void connectVertexBuffer( uint vertexBufferID );

    //////////////////////////////////////////////////////////////////////////

    void initializeDistributions();

    void initializeGeo();

    void collision();

    void postProcessing( char type );

    void computeMacroscopicQuantities();

    void scaleColorMap();

    void setGeo( uint xIdx, uint yIdx, char geo );

    void setGeo( uint xIdx1, uint yIdx1, uint xIdx2, uint yIdx2, char geo );

    void setGeoFloodFill( uint xIdx, uint yIdx, char geo );
    void setGeoFloodFillRecursion( uint xIdx, uint yIdx, char geo, charVecHost& hostGeo );

    void  setNu( float nu );
    float getNu();

    void  setU( float U );
    void  setV( float V );
	void  setAlpha(float alpha);
	void  setSpeed(float speed);
    void  setRefLength(uint ref);

    float getU();
    float getV();
	float getAlpha();
	float getSpeed();
    uint getRefLength();

    void setLBModel( char lbModel );
    char getLBModel();

    void setGeoMode( char geoMode );
    char getGeoMode();

    //////////////////////////////////////////////////////////////////////////

    void swap( floatVecPtr& lhs, floatVecPtr& rhs );

    uint c2i( uint xIdx, uint yIdx );

    D2Q9Ptr getDistPtr();
};

typedef std::shared_ptr<lbmSolver> lbmSolverPtr;

#endif