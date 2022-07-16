#pragma once

#include "./simple_hacky_types.hpp"

/*
// std::vector<point_t>
template <typename real>
inline collision_of_polyg cpoly_intersection__two_points(const fixedsize_polygon_with_side_metadata_t &poly1, const fixedsize_polygon_with_side_metadata_t &poly2);
*/

template <typename real>
inline real convex_polygon_area(const fixedsize_polygon_with_side_metadata_t &poly1)
{
    real area = 0;
    for (const side_meta_data_t &side : poly1)
    {
        real xyp = side.x0 * side.y1;
        real xyn = side.y0 * side.x1;
        area += xyp - xyn;
    }
    return area / 2.0;
}

// area from simple_hacky_polygp_t

// verbose, debug
template <typename real, bool verbose>
inline real convex_polygon_area2(const simple_hacky_polygp_t &poly1)
{
    real area = 0;
    {
        // C++23:
        // std::cout << "area(2):  " if consteval (verbose);
        // if constexpr (verbose) {
        if (verbose)
            std::cout << "area(2):  ";
        real last_x = poly1[poly1.size() - 1].first;
        real last_y = poly1[poly1.size() - 1].second;

        for (const pt2_t &pt : poly1)
        {
            real x = pt.first;
            real y = pt.second;
            if (verbose)
                std::cout << x << "," << y << ". " << std::endl;

            real xyp = last_x * y;
            real xyn = last_y * x;

            if (verbose)
                std::cout << "xy+= " << xyp << "-" << xyn << ". " << std::endl;

            area += xyp - xyn;
            if (verbose)
                std::cout << "area:" << area << std::endl;

            last_x = x;
            last_y = y;
        }
        if (verbose)
            std::cout << std::endl;
    }
    return area / 2.0;
}

