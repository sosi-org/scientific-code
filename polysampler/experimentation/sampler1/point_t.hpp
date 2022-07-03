export module point_t;

// #pragma once
// todo: int type
export struct point_t
{
    double x;
    double y;

public:
    std::string tostr() const
    {
        return "(" + std::to_string(this->x) + "," + std::to_string(this->y) + ")";
    }
}; // point_t;
