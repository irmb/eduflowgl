#ifndef WRITEFLOWFIELDTOVTKCOMMAND_H
#define WRITEFLOWFIELDTOVTKCOMMAND_H

#include "../Command.h"
#include <vector>


class writeFlowFieldToVTKCommand : public Command {
private:
    std::string filename;
    int nx;
    int ny;
    std::vector<float> velocityfield;
    std::vector<float> pressurefield;

public:
    writeFlowFieldToVTKCommand(const std::string& filename, int nx, int ny, std::vector<float> velocityfield, std::vector<float> pressurefield) : filename(filename), nx(nx), ny(ny), velocityfield(velocityfield), pressurefield(pressurefield) {}
    void execute() override;
    void undo() override;
    void redo() override;
};

#endif