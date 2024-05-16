#ifndef READSOLIDGEOMETRYCOMMAND_H
#define READSOLIDGEOMETRYCOMMAND_H

#include "../Command.h"
#include "../LBMSolver.h"
#include <vector>


class ReadSolidGeometryCommand : public Command {
private:
    LBMSolverPtr solver;
    std::string filePath;
    std::vector<char> laststate;

public:
    ReadSolidGeometryCommand(const std::string& path, LBMSolverPtr solver) : filePath(path), solver(solver) {}
    void execute() override;
    void undo() override;
    void redo() override;
};

#endif 
