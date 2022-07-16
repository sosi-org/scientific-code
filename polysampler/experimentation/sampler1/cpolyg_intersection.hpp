#pragma once

// for: intersect_lines_segment
#include "./line_intersection.hpp"
#include "./simple_hacky_types.hpp"

/*
template <typename real>
inline simple_hacky_polygp_t<real>
*/
/*
intersection between two convex polygons.

todo: add new augmented points, and add indices (?)
no: you just need the area
*/
template <typename real>
inline simple_hacky_polygp_t cpoly_intersection(const fixedsize_polygon_with_side_metadata_t &poly1, const fixedsize_polygon_with_side_metadata_t &poly2)
{
    return simple_hacky_polygp_t();
}

// result_thisfunc
struct collision_of_polyg
{
    // point1, side_id1 point2l side_id2
    // point_t point1, point2;
    // side_index_int_t side_1a, side_1b, side2a, side2b;
    point_t point[2];
    side_index_int_t side_1[2], side_2[2];
    // bool intersect;
    // no intersec
    // bool corner_on_corner;
    // bool corner_on_edge;
    /*
    enum class {no_intersect, two_intersect, corner_on_cronder, sorner_on_edge} result_situation;
    */
    // enum class {no_intersect, yes_intersection} situation;
    size_t count;

    void debug_print()
    {
        std::cout << count << ": ";
        for (int i = 0; i < count; ++i)
        {
            std::cout << side_1[i] << ":(" << point[i].x << "," << point[i].y << ")";
        }
        std::cout << count << ".";
    }
};
// collision
template <typename real>
inline collision_of_polyg cpoly_intersection__two_points(const fixedsize_polygon_with_side_metadata_t &poly1, const fixedsize_polygon_with_side_metadata_t &poly2)
{

    // C++20
    collision_of_polyg result{.count = 0};

    size_t ctr = 0;

    for (side_index_int_t s1 = 0; s1 < poly1.size(); ++s1)
    {
        const side_meta_data_t &side1 = poly1[s1];

        for (side_index_int_t s2 = 0; s2 < poly2.size(); ++s2)
        {

            const side_meta_data_t &side2 = poly2[s2];

            side_side_intersection_solution_t<real> is =
                intersect_lines_segment<real>(side1, side2);

            if (is.intersect)
            {
                result.point[ctr] = point_t{is.x, is.y};
                result.side_1[ctr] = s1;
                result.side_2[ctr] = s2;
                result.count = ctr + 1;
                ctr++;
                if (ctr >= 2)
                {
                    return result;
                }
            }
        }
    }
    return result;
    // return simple_hacky_polygp_t();
}

// followed by *.test.hpp
