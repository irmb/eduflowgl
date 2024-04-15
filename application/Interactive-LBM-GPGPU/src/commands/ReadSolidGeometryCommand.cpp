#include "ReadSolidGeometryCommand.h"


#include <SDL.h>




void readSolidGeometryFromBMP(const char* bmpFilePath, LBMSolverPtr solver) {
    std::cout << "Reading solid geometry from BMP file: " << bmpFilePath << std::endl;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        throw std::runtime_error("SDL_Init failed: " + std::string(SDL_GetError()));
    }

    SDL_Surface* bmpSurface = SDL_LoadBMP(bmpFilePath);
    if (!bmpSurface) {
        std::cerr << "Failed to load BMP file: " << SDL_GetError() << std::endl;
        SDL_Quit();
        throw std::runtime_error("Failed to load BMP file");
    }

    uint8_t* pixels = static_cast<uint8_t*>(bmpSurface->pixels);
    const int pitch = bmpSurface->pitch;
    std::vector<uint8_t> pixelData(pitch * bmpSurface->h);

   
        for (int y = 0; y < bmpSurface->h; y++) {
        memcpy(&pixelData[y * pitch], &pixels[y * pitch], pitch);
    }

    for (int y = 0; y < bmpSurface->h; y++) {
        for (int x = 0; x < bmpSurface->w; x++) {
            uint8_t pixelColor = pixelData[y * pitch + x * bmpSurface->format->BytesPerPixel];

            if (pixelColor != 0) {
                solver->setGeo(x, 600 - y, GEO_SOLID); 
                
            }
        }
    }
    
    

    

    SDL_FreeSurface(bmpSurface);
    SDL_Quit();
    std::cout << "Finished reading solid geometry from BMP file." << std::endl;
}

void ReadSolidGeometryCommand::execute() {
    try
    {
        laststate = solver->getgeoData();
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error occurred: " << e.what() << std::endl;
    }
    
    
    try {
        readSolidGeometryFromBMP(filePath.c_str(), solver);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void ReadSolidGeometryCommand::undo() {
try {
    solver->setgeoData(laststate);
    } catch (const std::exception& e) {
    
    std::cerr << "Error occurred: " << e.what() << std::endl;
}

   
    

   
}

void ReadSolidGeometryCommand::redo() {
    // Redo logic here if needed
}
