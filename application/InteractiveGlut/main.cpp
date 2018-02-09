#include <vector>
#include <chrono>
#include <iostream>
#include <memory>

#include "lbm.h"
#include "Visualizer.h"

//////////////////////////////////////////////////////////////////////////

typedef unsigned int uint;

//////////////////////////////////////////////////////////////////////////

const int NX = 300*5;
const int NY = 100*5;

const uint pxPerNode = 1;

const uint timeStepsPerFrame = 50;

const float U = 0.025f;
const float V = 0.000f;

const float nu    = U * NY/8 / 100;
const float omega = 2.0f / ( 6.0f * nu + 1.0f ); // for Re 100

//////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    lbmSolverPtr solver = std::make_shared<lbmSolver>( NX, NY, omega, U, V );

    solver->initializeDistributions();

    //////////////////////////////////////////////////////////////////////////

    Visualizer::initialize( argc, argv, NX, NY, pxPerNode, timeStepsPerFrame, solver );

    Visualizer::installShaders();

    Visualizer::generateVertices();

    Visualizer::generateElements();

    solver->connectVertexBuffer( Visualizer::getVertexBufferID() );

    //////////////////////////////////////////////////////////////////////////

    for( int x = -NY/8; x <= NY/8; x++ ){
        for( int y = -NY/8; y <= NY/8; y++ ){
            if( sqrt( x * x + y * y ) > NY/8 ) continue;
            solver->setGeo(NY/2 + x, NY/2 + y,1);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    Visualizer::run();

    return 0;
}


