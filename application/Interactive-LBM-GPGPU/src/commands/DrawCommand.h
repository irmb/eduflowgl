#ifndef DRAWCOMMAND_H
#define DRAWCOMMAND_H

#include "../Command.h"
#include "../LBMSolver.h"
#include <vector>



class DrawCommand : public Command {
private:
    LBMSolverPtr solver;
    uint xIdx;
    uint yIdx;
    uint xIdxLast;
    uint yIdxLast;
    char geo;
    int com;
    std::vector<char> laststate;
    std::vector<char> laststate2;
    
public:
    DrawCommand(LBMSolverPtr solver, uint xIdx, uint yIdx,uint xIdxLast,uint yIdxLast, char geo, int com) : solver(solver), xIdx(xIdx), yIdx(yIdx), xIdxLast(xIdxLast), yIdxLast(yIdxLast), geo(geo), com(com) {}
    void execute() override;
    void undo() override;
    void redo() override;
};

#endif