#include "Commands.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

#include <SDL.h>



void Commands::writeFlowFieldToVTK(const std::string& filename, int nx, int ny, std::vector<float> velocity, std::vector<float> pressure) {
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

void Commands::readSolidGeometryFromBMP(const char* bmpFilePath, LBMSolverPtr solver)
{
    std::cout << "Reading solid geometry from BMP file: " << bmpFilePath << std::endl;

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        throw std::runtime_error("SDL_Init failed: " + std::string(SDL_GetError()));
    }

    SDL_Surface* bmpSurface = SDL_LoadBMP(bmpFilePath);
    if (!bmpSurface)
    {
        std::cerr << "Failed to load BMP file: " << SDL_GetError() << std::endl;
        SDL_Quit();
        throw std::runtime_error("Failed to load BMP file");
    }

    
    uint8_t* pixels = static_cast<uint8_t*>(bmpSurface->pixels);

    
    const int pitch = bmpSurface->pitch;
    std::vector<uint8_t> pixelData(pitch * bmpSurface->h);
    for (int y = 0; y < bmpSurface->h; y++)
    {
        memcpy(&pixelData[y * pitch], &pixels[y * pitch], pitch);
    }

    for (int y = 0; y < bmpSurface->h; y++)
    {
        for (int x = 0; x < bmpSurface->w; x++)
        {
            
            uint8_t pixelColor = pixelData[y * pitch + x * bmpSurface->format->BytesPerPixel];

            
            // std::cout << "Pixel at (" << x << ", " << y << ") has color: " << static_cast<int>(pixelColor) << std::endl;

            if (pixelColor != 0)
            {
                
                solver->setGeo(x, 600 - y, GEO_SOLID);
            }
        }
    }

    
    SDL_FreeSurface(bmpSurface);
    SDL_Quit();
    std::cout << "Finished reading solid geometry from BMP file." << std::endl;
}

float Commands::compareVelocityProfiles(const std::vector<float>& simulationData, const std::vector<float>& benchmarkData) {
   
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



    std::vector<float> Commands::readVelocityProfileFromVTK(const std::string& filename) {
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


