#ifndef COMMANDS_H
#define COMMANDS_H
#include <cstdint>
#include <string>
#include <vector>
#include "LBMSolver.h"



class Commands {
public:
    static void writeFlowFieldToVTK(const std::string& filename, int nx, int ny, std::vector<float> velocity, std::vector<float> pressure);
    static void readSolidGeometryFromBMP(const char* bmpFilePath, LBMSolverPtr solver);
    static std::vector<float> readVelocityProfileFromVTK(const std::string& filename);
    static float compareVelocityProfiles(const std::vector<float>& simulationData, const std::vector<float>& benchmarkData);
};

#endif // Commands_H
