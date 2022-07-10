#pragma once

/*
Line-segment intersection
*/

#include "./side_meta_data_t.hpp"

template <typename real>
struct side_side_intersection_solution_t
{
    bool intersect;
    // real condition_number; // ie do not intersect in case of parallel. or too far.
    real x;
    real y;
};

/*
template <typename real>
inline side_side_intersection_solution_t<real> intersect_lines(const side_meta_data_t &side1, const side_meta_data_t &side2) {
    // todo: raw formula, returning nu_u,nu_v,denom
    // without the boolean part

    evolution:
    (side,side) -> (numerator_x,numerator_y,denom) -> side_side_intersection_solution_t
}
*/
template <typename real>
inline side_side_intersection_solution_t<real> intersect_lines(const side_meta_data_t &side1, const side_meta_data_t &side2)
{

    real x1 = side1.x0;
    real y1 = side1.y0;
    real x2 = side1.x1;
    real y2 = side1.y1;

    real x3 = side2.x0;
    real y3 = side2.y0;
    real x4 = side2.x1;
    real y4 = side2.y1;

    real numerator_x = x1 * x3 * y2 - x1 * x3 * y4 - x1 * x4 * y2 + x1 * x4 * y3 - x2 * x3 * y1 + x2 * x3 * y4 + x2 * x4 * y1 - x2 * x4 * y3;
    real numerator_y = x1 * y2 * y3 - x1 * y2 * y4 - x2 * y1 * y3 + x2 * y1 * y4 - x3 * y1 * y4 + x3 * y2 * y4 + x4 * y1 * y3 - x4 * y2 * y3;
    real denom = x1 * y3 - x1 * y4 - x2 * y3 + x2 * y4 - x3 * y1 + x3 * y2 + x4 * y1 - x4 * y2;

    // typical extents of the values: -0.585 , -0.325 , -0.65

    /*
    std::cout << "numerator_x:" << numerator_x << std::endl;
    std::cout << "numerator_y:" << numerator_y << std::endl;
    std::cout << "numerator_d:" << denom << std::endl;
    */
    /*
    <circle cx="" cy="" r="0.05" fill="red" />
    */
    return side_side_intersection_solution_t<real>{
        true,
        numerator_x / denom,
        numerator_y / denom,
    };
}

#include <cmath>
#include <algorithm>

// static

// false
inline side_side_intersection_solution_t<double> null_intersection{false, 0, 0};


template <typename real>
inline side_side_intersection_solution_t<real> intersect_lines_segment(const side_meta_data_t &side1, const side_meta_data_t &side2)
{

    // also: get the denom
    real adx1 = std::abs(side1.x0 - side1.x1);
    real ady1 = std::abs(side1.y0 - side1.y1);
    real adx2 = std::abs(side2.x0 - side2.x1);
    real ady2 = std::abs(side2.y0 - side2.y1);

    /* bounding box method */
    // units: in pixels
    constexpr real ε = 0.00000000001;
    // more permissive:
    constexpr real ε2 = 0.0000001;

    auto [minx1, maxx1] = std::minmax(side1.x0, side1.x1);
    auto [minx2, maxx2] = std::minmax(side2.x0, side2.x1);
    if (minx1 > maxx2 + ε)
    {
        return null_intersection;
    }
    if (minx2 > maxx1 + ε)
    {
        return null_intersection;
    }
    /*
    if (minx1 > maxx2 + ε) {
        return null_intersection;
    }
    if (miny2 > maxy1 + ε) {
        return null_intersection;
    }
    */

    auto [miny1, maxy1] = std::minmax(side1.y0, side1.y1);
    auto [miny2, maxy2] = std::minmax(side2.y0, side2.y1);
    if (miny1 > maxy2 + ε)
    {
        return null_intersection;
    }
    if (miny2 > maxy1 + ε)
    {
        return null_intersection;
    }

    side_side_intersection_solution_t<real> xy =
        intersect_lines<real>(side1, side2);

    if (minx1 > xy.x + ε2)
    {
        return null_intersection;
    }
    if (minx2 > xy.x + ε2)
    {
        return null_intersection;
    }
    if (miny1 > xy.y + ε2)
    {
        return null_intersection;
    }
    if (miny2 > xy.y + ε2)
    {
        return null_intersection;
    }
    // This is not enough. It will not work well if one of the lines is paralllel (or almost parallel) to one of the axes.
    // next: more precise: use two perpendicular planes. Parts of solution 2,3 below.

    // Idea: Alternative solution: project line B to line A's base space.
    //       Weakness: if almost parallel, not very efficient.
    //                 but on the other hand, most likely they will not cross.
    //                 try projecting both onto perp-space. (solution 3)

    return xy;
}
