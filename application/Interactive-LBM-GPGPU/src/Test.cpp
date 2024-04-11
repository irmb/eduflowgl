#include "Test.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <string>
#include <cmath>
#include <cstdlib>
#include "LBMSolver.h"





float compareVelocityProfiles(const std::vector<float>& simulationData, const std::vector<float>& benchmarkData) {
   
    if (simulationData.size() != benchmarkData.size()) {
        std::cerr << "Error: Data sizes do not match." << std::endl;
        return 10.0;
    }

    
    float rmse = 0.0;
    for (size_t i = 0; i < simulationData.size(); ++i) {
        float diff = simulationData[i] - benchmarkData[i];
        rmse += diff * diff;
    }
    rmse = sqrt(rmse / simulationData.size());

    std::cout << "Root Mean Square Error (RMSE): " << rmse << std::endl;

    return rmse;
}



    std::vector<float> readVelocityProfileFromVTK(const std::string& filename) {
    std::vector<float> velocityProfile;

    std::ifstream inputFile(filename);

    
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return velocityProfile;
    }

    std::string line;
    bool dimensionsFound = false;
    while (std::getline(inputFile, line)) {
        if (!dimensionsFound && line.find("DIMENSIONS") != std::string::npos) {
            std::istringstream iss(line);
            std::string dummy;
            int nx, ny;
            iss >> dummy >> nx >> ny;
            velocityProfile.resize(nx * ny);
            dimensionsFound = true;
        } else if (line.find("VECTORS Velocity") != std::string::npos) {
            
            int count = 0;
            float vx, vy, vz;
            while (inputFile >> vx >> vy >> vz && count < velocityProfile.size()) {
                velocityProfile[count] = vx; 
                ++count;
            }
            break; 
        }
    }

    inputFile.close();

    return velocityProfile;
}
int Test::runTest(float threshold) {
    
    std::cout << "Running test...\n";
   
    int NX = 1000; 
    int NY = 600;
    int Lref=(12*NY)/30;
    const float U = 0.08f; 
    const float V = 0.0f;
    const float nu = 0.1;
    const float omega = 2.0f / (6.0f * nu + 1.0f);
    const uint timeStepsPerFrame = 300;

    
    LBMSolverPtr solver = std::make_shared<LBMSolver>(NX, NY, omega, U, V);
    solver->initializeDistributions();
    solver->initializeGeo();
    solver->setRefLength(NY);
    int r = Lref/2;

    for( int x = -r; x <= r; x++ ){
        for( int y = -r; y <= r; y++ ){
            if( sqrt( x * x + y * y ) > r ) continue;
            solver->setGeo(1.9*r + x, NY/2 + y+50,GEO_SOLID);
        }
    }
    
    for (int j = 0; j < 19; j++)
    {
   
    
        for( int i = 0; i < timeStepsPerFrame; i++ ){
        solver->collision();
        }
        solver->computeMacroscopicQuantities();
         
}
        
   
    float rmse = compareVelocityProfiles(solver->getVelocityData(),readVelocityProfileFromVTK("benchmark_data.vtk"));
    if (rmse < threshold)
    {
        std::cout << "RMSE within threshold Test passed ..."<< std::endl;
        return 0;
    }
    else
    std::cout << "RMSE greater than threshold Test Failed ..."<< std::endl;
    return 1;
    
   
    
}