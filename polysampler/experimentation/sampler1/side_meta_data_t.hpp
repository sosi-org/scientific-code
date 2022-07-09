#pragma once

#include <array>

import point_t;
import tesselation_t;
// #include "tesselation_t.hpp" // for points_t only

struct side_meta_data_t
{
    double x0, y0;
    double dx, dy; // x1-x0, y1-y0

    side_meta_data_t(const point_t &p0, const point_t &p1)
    {
        this->x0 = p0.x;
        this->y0 = p0.y;
        this->dx = p1.x - p0.x;
        this->dy = p1.y - p0.y;
    }
};

// std::array<side_meta_data_t>
std::vector<side_meta_data_t> convert(const tesselation_t &tesselation,
                                      const points_t &points)
{
    // return std::array<side_meta_data_t, 3>
    //return std::vector<side_meta_data_t>(tesselation.begin(), tesselation.end());
    return std::vector<side_meta_data_t>();
}
