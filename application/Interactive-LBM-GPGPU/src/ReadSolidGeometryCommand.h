#ifndef READSOLIDGEOMETRYCOMMAND_H
#define READSOLIDGEOMETRYCOMMAND_H

#include "Command.h"
#include "LBMSolver.h"



class ReadSolidGeometryCommand : public Command {
private:
    LBMSolverPtr solver;
    std::string filePath;
    char* laststate = new char[600000];

public:
    ReadSolidGeometryCommand(const std::string& path, LBMSolverPtr solver) : filePath(path), solver(solver) {}
    void execute() override;
    void undo() override;
    void redo() override;
};

#endif // READSOLIDGEOMETRYCOMMAND_H
