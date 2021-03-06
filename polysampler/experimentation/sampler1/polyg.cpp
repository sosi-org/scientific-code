export module polyg;
// #pragma once

#include <vector>
#include <iostream>

// import side_meta_data_t;
import side_meta_data_t;
import tesselation_t;
// std.experimental.vector

export typedef std::vector<side_point_t> vertiex_indices_t; // vertices; // without coords, just int, refering to the coords index

// const std::vector<side_point_t> &points_indices_ref;  = vertices
//  const vertiex_indices_t &vertiex_indices,

// polygon:

// std::array<side_meta_data_t>
// typedef std::unique_ptr<side_meta_data_t[]> fixedsize_polygon_with_side_metadata_t;
export typedef std::vector<side_meta_data_t> fixedsize_polygon_with_side_metadata_t;

// fixedsize_polygon_with_side_metadata_t = polygon

// todo: rename: fixedsize_polygon_with_side_metadata_t ->? polyg_fixedsize_side_metadata_t

/*
// only corners
export void debug_print_xy0(const fixedsize_polygon_with_side_metadata_t &poly)
{
    std::cout << poly.size() << ":";
    for (auto v : poly)
    {
        std::cout << "(" << v.x0 << "," << v.y0 << ") ";
    }
    // todo: how to avoid such repeat?
}
*/
