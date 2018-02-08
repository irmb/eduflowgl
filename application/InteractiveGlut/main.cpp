#include <vector>
#include <chrono>
#include <iostream>
#include <memory>

#include "lbm.h"
#include "Visualizer.h"

//////////////////////////////////////////////////////////////////////////

typedef unsigned int uint;

//////////////////////////////////////////////////////////////////////////

const uint NX = 1600;
const uint NY = 1000;

const uint PXPP = 1;

//////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    lbmSolverPtr solver = std::make_shared<lbmSolver>( NX, NY );

    solver->initializeDistributions();

    //////////////////////////////////////////////////////////////////////////

    Visualizer::initialize( argc, argv, NX, NY, PXPP, solver );

    Visualizer::installShaders();

    Visualizer::generateVertices();

    Visualizer::generateElements();

    solver->connectVertexBuffer( Visualizer::getVertexBufferID() );

    //////////////////////////////////////////////////////////////////////////

    Visualizer::run();

    return 0;
}


