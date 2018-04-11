#include "StopWatch.h"

StopWatch::StopWatch()
{
    this->reset();
}

StopWatch::timePoint StopWatch::reset()
{
    return this->startTime = std::chrono::high_resolution_clock::now();
}

StopWatch::Seconds StopWatch::getElapsedSeconds()
{
    return std::chrono::duration_cast<std::chrono::microseconds>( this->now() - startTime ).count() / 1000000.0;
}

StopWatch::MilliSeconds StopWatch::getElapsedMilliSeconds()
{
    return std::chrono::duration_cast<std::chrono::microseconds>( this->now() - startTime ).count() / 1000.0;;
}

StopWatch::timePoint StopWatch::now()
{
    return std::chrono::high_resolution_clock::now();
}
