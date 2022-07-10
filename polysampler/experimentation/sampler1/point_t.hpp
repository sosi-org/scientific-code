export module point_t;
// export module point_t.hpp;
//  export module point_t;
#include <iostream>

#include <sstream>
#include <iomanip>

// #export module point_t;
// #pragma once

// #pragma once
// todo: int type
// export
export struct point_t
{
    double x;
    double y;

public:
    std::string tostr() const
    {
        std::ostringstream oss;
        oss << "(";
        oss << std::setprecision(8) << std::noshowpoint << this->x;
        oss << ",";
        oss << std::setprecision(8) << std::noshowpoint << this->y;
        oss << ")";
        return oss.str();
        // return "(" + std::to_string(this->x) + "," + std::to_string(this->y) + ")";
    }
}; // point_t;
