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
