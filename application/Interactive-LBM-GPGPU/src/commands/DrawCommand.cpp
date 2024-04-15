#include "DrawCommand.h"




void DrawCommand::execute() {
    laststate = solver->getgeoData();
    if (com == 1) {
       
        solver->setGeo(xIdxLast, yIdxLast, xIdx, yIdx, geo);
    } else if (com == 2) {
       
        solver->setGeoFloodFill(xIdx, yIdx, geo);
    }
    else if (com == 3)
    {
        solver->setGeo(xIdx, yIdx, geo);
    }
    
}



void DrawCommand::undo() {
solver->setgeoData(laststate);
char* laststate = new char[600000];
   
    

    
}

void DrawCommand::redo() {
    // Redo logic here if needed
}
