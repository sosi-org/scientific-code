export module side_meta_data_t;
// #pragma once

#include <array>

import point_t;
import tesselation_t;
// #include "tesselation_t.hpp" // for points_t only

/*
    Datastructure fot storing cacehed augmented data about a side of a polygon
*/
// todo: template<real>
export
class side_meta_data_t
{
public: // not good
    double x0, y0;
    double dx, dy; // x1-x0, y1-y0
    double x1, y1;

public:
    side_meta_data_t(const point_t &p0, const point_t &p1)
    {
        this->x0 = p0.x;
        this->y0 = p0.y;
        this->dx = p1.x - p0.x;
        this->dy = p1.y - p0.y;

        this->x1 = p1.x;
        this->y1 = p1.y;
    }
    /*
    side_meta_data_t() = default;
    side_meta_data_t(side_meta_data_t const&) = default;
    side_meta_data_t& operator=(side_meta_data_t&) = default;
    */
    /*
    // side_meta_data_t other)
    side_meta_data_t& operator=(const side_meta_data_t &other)
    {
        this->x0 = other.x0;
        this->y0 = other.y0;
        this->dx = other.dx;
        this->dy = other.dy;
        return *this;
    }
    */
};

// import std.vector;
#include <vector>


fixedsize_side_metadata_t convert(const tesselation_t &tesselation,
                                      const points_t &points)
{
    // return std::array<side_meta_data_t, 3>
    // return std::vector<side_meta_data_t>(tesselation.begin(), tesselation.end());
    return std::vector<side_meta_data_t>();
}
