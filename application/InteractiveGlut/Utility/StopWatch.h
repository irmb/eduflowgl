#ifndef StopWatch_h
#define StopWatch_h

#include <chrono>
#include <memory>

class StopWatch {

public:
    
typedef double Seconds;
typedef double MilliSeconds;
typedef std::chrono::high_resolution_clock::time_point timePoint;

private:
    timePoint startTime;

public:
    StopWatch();

    timePoint    reset();

    Seconds      getElapsedSeconds();
    MilliSeconds getElapsedMilliSeconds();

    timePoint    now();
};

typedef std::shared_ptr<StopWatch> StopWatchPtr;

#endif