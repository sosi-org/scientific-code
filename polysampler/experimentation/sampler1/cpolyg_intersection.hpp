#pragma once

#include <vector>

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
                // good side condition: side_1[0] < side_1[1]
                result.side_2[ctr] = s2;
                result.count = ctr + 1;
                ctr++;
                if (ctr >= 2)
                {
                    assert(result.side_1[0] < result.side_1[1]);
                    return result;
                }
            }
        }
    }
    return result;
    // return simple_hacky_polygp_t();
}

simple_hacky_polygp_t to_simple_hacky_polygp_t(const fixedsize_polygon_with_side_metadata_t &poly)
{
    simple_hacky_polygp_t pp;
    for (const side_meta_data_t &s : poly)
    {
        pt2_t pt{s.x0, s.y0};
        // ignore: .dx, .dy, .x1, .y1

        pp.push_back(pt);
    }
    return pp;
}

/*
The type:
 `fixedsize_polygon_with_side_metadata_t`
 was originally supposed to be a type for which intersections are convenient.

*/

//enum class {no_erase, erase_A, erase_B};

// can be executed in next vectorized round
// asymmetric: always use the first poly as basis.
// future: se can extract the completemnet (A - B) but we need the intersection only now (A ∩ B).
template <typename real>
inline simple_hacky_polygp_t
cpoly_intersection__complete_poly(const fixedsize_polygon_with_side_metadata_t &poly1, const fixedsize_polygon_with_side_metadata_t &poly2,
bool dont_erase, bool erase_between)
{
    simple_hacky_polygp_t rpoly; // keep empty hull

    collision_of_polyg collision = cpoly_intersection__two_points<real>(poly1, poly2);
    constexpr int TWO = 2; // magical number
    if (collision.count == TWO)
    {
        // point_t point[2];
        // side_index_int_t side_1[2], side_2[2];
        // take second polygon

        side_index_int_t new_point_indices[2];

        for (int collidx = 0; collidx < TWO; collidx++)
        {

            // what can be do with `poly1`?
            // in fact, we can refer back to its original. maybe a pointer or an integer index in the tessellation?
            // but in this algorithm, we really do not need to add these vertices back. we just need to create some polygon to calculate the area with
            // so, let's see what structures does area() accept.

            // simple_hacky_polygp_t rpoly
            //  how to move from fixedsize_polygon_with_side_metadata_t to simple_hacky_polygp_t ?

            // I am on the right track: I ended up to the same signature/declaration as another one on top of this file
            // simple_hacky_polygp_t rpoly = to_simple_hacky_polygp_t(poly1);
            if (collidx == 0)
            {
                rpoly = to_simple_hacky_polygp_t(poly1);
            }

            // vector<pt2_t>
            // simple_hacky_polygp_t

            side_index_int_t i1 = collision.side_1[collidx];
            // i2, will not be used, because we are building on top of poly1
            side_index_int_t i2 = collision.side_2[collidx];
            point_t new_point = collision.point[collidx];

            side_index_int_t i1next = i1 + 1;
            // It is between i1 and i1next=i1+1 .
            auto position1 = rpoly.begin() + i1next;
            // std::vector::insert(position1, pt2_t{new_point.x, new_point.y} );
            rpoly.insert(position1, pt2_t{new_point.x, new_point.y});

            new_point_indices[collidx] = i1next;

            // It is between i1 and i1+1?
            for (int i = collidx + 1; i < TWO; i++)
            {
                // needs to be tested and re-thought
                if (collision.side_1[i] >= i1next) // if on the right side (shifted part) in the vector<>
                    collision.side_1[i]++;
                /*
                // in fact side_2 will not be used
                if (collision.side_2[i] >= i1next)
                    collision.side_2[i]++;
                */
            }
            if (build.debug)
            {
                std::cout << "in progress:";
                debug_print(rpoly);
                std::cout << std::endl;
            }
        }
        // challenge: which side to keep?
        // collision.side_1[0],collision.side_1[1]
        // new_point_indices[0], [1]
        // slow method


        if (build.debug)
        {
            if (new_point_indices[1] < new_point_indices[0])
            {
                std::cout << "smaller" << new_point_indices[0] << "<" << new_point_indices[1] << std::endl;
            }
            for (int i = new_point_indices[0]; i < new_point_indices[1]; i++)
            {
                std::cout << i << "";
            }
            std::cout << std::endl;
        }
        if (!dont_erase) {
        // note: assert(side_1[0] < side_1[1]);
        assert(new_point_indices[0] < new_point_indices[1]);
        assert(new_point_indices[0]+1 <= new_point_indices[1]-1); // becauwe we increased the second one
        if (erase_between) {
            size_t before_size = rpoly.size();
            side_index_int_t from = new_point_indices[0]+1;
            side_index_int_t to = new_point_indices[1]-1;
            assert(from <= to); // proved
            rpoly.erase(rpoly.begin() + from, rpoly.begin() + to+1 );
            if (build.debug)
            {
                std::cout << "debugA:"
                          << before_size << "-"
                          << rpoly.size()
                          << "=="
                          << to
                          << "-" << from
                          << "+1"
                          << std::endl;
               // (0,0) (1,0) *(1,0.5) (1,1) *(0.5,1) (0,1)
               // debug:6-4==4-2-1
            }
            assert( before_size-rpoly.size() == to - from + 1);
        } else {
            size_t before_size = rpoly.size();
            side_index_int_t frombegin_to = new_point_indices[0]-1;
            side_index_int_t from_toend = new_point_indices[1]+1;
            assert(frombegin_to < from_toend); // how to prove?
            // in worst case, there is one node in the angle
            rpoly.erase(rpoly.begin() + from_toend, rpoly.end());  // not inclusive for from_toend
            rpoly.erase(rpoly.begin(), rpoly.begin() + frombegin_to+1 ); // not inclusive for frombegin_to
            if (build.debug)
            {
                std::cout << "debugB:"
                          //<< before_size << "-"
                          << rpoly.size()
                          << "=="
                          << new_point_indices[1]   // 4
                          << "-" << new_point_indices[0] // 2
                          << "+1"
                          << std::endl;
               // (0,0) (1,0) *(1,0.5) (1,1) *(0.5,1) (0,1)
               // debug:6-4==4-2-1

                // in progress:6:(0,0) (1,0) (1,0.5) (1,1) (0.5,1) (0,1)
                // debugB:6-5==4-2-1

            }
            assert(
                rpoly.size() ==
                new_point_indices[1]-new_point_indices[0]+1
            );
        }
        }
        return rpoly;
    }
    // empty
    std::cout << "todo: I dont know if empty, or all of it" << std::endl;
    return simple_hacky_polygp_t{};
}

// followed by *.test.hpp
