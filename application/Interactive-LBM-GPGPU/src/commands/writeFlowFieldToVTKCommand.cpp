#include "writeFlowFieldToVTKCommand.h"
#include <iostream>     
#include <fstream>      
#include <vector>       
#include <string>       
#include <stdexcept>




void writeFlowFieldToVTK(const std::string& filename, int nx, int ny, std::vector<float> velocity, std::vector<float> pressure) {
    std::ofstream outputFile(filename);

   
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    
    outputFile << "# vtk DataFile Version 3.0" << std::endl;
    outputFile << "LBM Flow Field Data" << std::endl;
    outputFile << "ASCII" << std::endl;
    outputFile << "DATASET STRUCTURED_POINTS" << std::endl;
    outputFile << "DIMENSIONS " << nx << " " << ny << " 1" << std::endl;
    outputFile << "ORIGIN 0 0 0" << std::endl;
    outputFile << "SPACING 1 1 1" << std::endl;
    outputFile << "POINT_DATA " << nx * ny << std::endl;
    outputFile << "VECTORS Velocity float" << std::endl;

   
  
    for (int yIdx = 0; yIdx < ny; ++yIdx) {
        for (int xIdx = 0; xIdx < nx; ++xIdx) {
            int index = yIdx * nx + xIdx;
            outputFile << velocity[index] << " 0 0" << std::endl; 
        }
    }

   
    outputFile << "SCALARS Pressure float 1" << std::endl;
    outputFile << "LOOKUP_TABLE default" << std::endl;

    for (int yIdx = 0; yIdx < ny; ++yIdx) {
        for (int xIdx = 0; xIdx < nx; ++xIdx) {
            int index = yIdx * nx + xIdx;
            outputFile << pressure[index] << std::endl;
        }
    }

   
    outputFile.close();

    std::cout << "Flow field data has been written to " << filename << std::endl;

}

void writeFlowFieldToVTKCommand::execute() {
    try
    {
        writeFlowFieldToVTK(filename, nx, ny, velocityfield, pressurefield);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error occurred: " << e.what() << std::endl;
    }
    
}

void writeFlowFieldToVTKCommand::undo() {

    
}

void writeFlowFieldToVTKCommand::redo() {
    
}