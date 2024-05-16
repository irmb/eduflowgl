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

laststate2 = solver->getgeoData();
solver->setgeoData(laststate);

   
    

    
}

void DrawCommand::redo() {
    solver->setgeoData(laststate2);
}
