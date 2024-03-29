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

// declaration only
bool is_inside_poly(const fixedsize_polygon_with_side_metadata_t &poly, const point_t &);

// erase_inside
// delete [a,...,b], a<b
side_index_int_t erase_between(simple_hacky_polygp_t &rpoly, side_index_int_t new_point_indices[2])
{
    size_t before_size = rpoly.size();
    side_index_int_t from = new_point_indices[0] + 1;
    side_index_int_t to = new_point_indices[1] - 1;
    assert(from <= to); // proved
    rpoly.erase(rpoly.begin() + from, rpoly.begin() + to + 1);
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
    assert(before_size - rpoly.size() == to - from + 1);
    // next insert index
    return from;
}

// delete [..., a, b, ...], a<b
side_index_int_t erase_outwards(simple_hacky_polygp_t &rpoly, side_index_int_t new_point_indices[2])
{
    size_t before_size = rpoly.size();
    side_index_int_t frombegin_to = new_point_indices[0] - 1;
    side_index_int_t from_toend = new_point_indices[1] + 1;
    assert(frombegin_to < from_toend); // how to prove?
    // in worst case, there is one node in the angle
    rpoly.erase(rpoly.begin() + from_toend, rpoly.end());         // not inclusive for from_toend
    rpoly.erase(rpoly.begin(), rpoly.begin() + frombegin_to + 1); // not inclusive for frombegin_to
    if (build.debug)
    {
        std::cout << "debugB:"
                  //<< before_size << "-"
                  << rpoly.size()
                  << "=="
                  << new_point_indices[1]        // 4
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
        new_point_indices[1] - new_point_indices[0] + 1);
    // next insert index
    return rpoly.size();
}

/*
The type:
 `fixedsize_polygon_with_side_metadata_t`
 was originally supposed to be a type for which intersections are convenient.

*/

enum class erasing_mod_t
{
    no_erase,
    erase_between,
    erase_outside,
    erase_auto
};

// can be executed in next vectorized round
// asymmetric: always use the first poly as basis.
// future: se can extract the completemnet (A - B) but we need the intersection only now (A ∩ B).
template <typename real>
inline simple_hacky_polygp_t
cpoly_intersection__complete_poly(const fixedsize_polygon_with_side_metadata_t &poly1, const fixedsize_polygon_with_side_metadata_t &poly2,
                                  erasing_mod_t mode)
{
    simple_hacky_polygp_t rpoly; // keep empty hull

    collision_of_polyg collision = cpoly_intersection__two_points<real>(poly1, poly2);
    constexpr int TWO = 2; // magical number
    if (collision.count == TWO)
    {
        // point_t point[2];
        // side_index_int_t side_1[2], side_2[2];
        // take second polygon

        side_index_int_t new_point_indices1[2];
        side_index_int_t new_point_indices2[2];

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

            new_point_indices1[collidx] = i1next;
            new_point_indices2[collidx] = i2; // mirrors the collision.side_2

            // It is between i1 and i1+1?
            for (int i = collidx + 1; i < TWO; i++)
            {
                // needs to be tested and re-thought
                if (collision.side_1[i] >= i1next) // if on the right side (shifted part) in the vector<>
                    collision.side_1[i]++;
                /*
                //not changed
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

        // for poly2:
        if (new_point_indices2[0] > new_point_indices2[1])
        {
            auto swap_temp = new_point_indices2[0];
            new_point_indices2[0] = new_point_indices2[1];
            new_point_indices2[1] = swap_temp;
        }
        assert(new_point_indices2[0] < new_point_indices2[1]);
        // lines: (new_point_indices2[0], new_point_indices2[0]+1) , (new_point_indices2[1],new_point_indices2[1]+1)
        side_index_int_t a = new_point_indices2[0], b = new_point_indices2[0] + 1, c = new_point_indices2[1], d = new_point_indices2[1] + 1;
        // inner_range: new_point_indices2[0]+1 , new_point_indices2[1]
        // either: (b,c)  or (d,a) circular
        side_index_int_t midpoint2 = (b + c) / 2;

        // in poly1:
        // challenge: which side to keep?
        // collision.side_1[0],collision.side_1[1]
        // new_point_indices1[0], [1]
        // slow method

        if (build.debug)
        {
            if (new_point_indices1[1] < new_point_indices1[0])
            {
                std::cout << "smaller" << new_point_indices1[0] << "<" << new_point_indices1[1] << std::endl;
            }
            for (int i = new_point_indices1[0]; i < new_point_indices1[1]; i++)
            {
                std::cout << i << "";
            }
            std::cout << std::endl;
        }

        simple_hacky_polygp_t rpoly2;

        if (mode == erasing_mod_t::erase_auto)
        {
            // get a point in between
            side_index_int_t midpoint_index =
                (new_point_indices1[0] + new_point_indices1[1]) / 2;
            // see below for proof
            assert(new_point_indices1[0] + 1 <= new_point_indices1[1] - 1);

            /*
            Proof:

            new_point_indices1[0] + 1 <= new_point_indices1[1] - 1
            // n := new_point_indices1
            n[0] + 1 <= n[1] - 1
            n[0] + 2 <= n[1]
            n[0] + 2 + δ == n[1], ∃ δ >= 0
            n[0]*2 + 2 + δ == n[1] +n[0] , ∃δ≥0
            n[0] + ⌊(2 + δ)/2⌋ == (n[1] +n[0])/2 , ∃δ≥0
            n[0] + 1 + ⌊δ/2⌋ == midpoint_index ,  ∃δ≥0
            n[0] < midpoint_index
            new_point_indices1[0] < midpoint_index
            Q.E.D


            n[0] + 2 + δ == n[1] , ∃ δ ≥ 0
            n[0] + n[1] + 2 + δ == n[1]*2  , ∃ δ ≥ 0
            (n[0] + n[1])/2 + (2 + δ)/2 == n[1]*2/2  , ∃ δ ≥ 0
            midpoint_index + 1 + ⌊δ/2⌋ == n[1]  , ∃ δ ≥ 0
            midpoint_index < n[1]
            midpoint_index < new_point_indices1[1]
            Q.E.D
            */

            // note: strict inequality (unequal)
            assert(midpoint_index > new_point_indices1[0]);
            assert(midpoint_index < new_point_indices1[1]);

            const auto &midpoint = rpoly[midpoint_index];
            // now what?
            // too slow
            // also erasing and insering in the vector is too much waster of CPU
            bool is_inside = is_inside_poly(poly2, point_t{.x = midpoint.first, .y = midpoint.second});
            if (is_inside)
            {
                // keep midpoint
                mode = erasing_mod_t::erase_outside;
            }
            else
            {
                mode = erasing_mod_t::erase_between;
            }

            /*
            side_index_int_t midpoint_index2 =
                (new_point_indices2[0] + new_point_indices2[1]) / 2;
            assert(new_point_indices2[0] < new_point_indices2[1]);
            */

            //such waste of CPU.
            simple_hacky_polygp_t poly2t = to_simple_hacky_polygp_t(poly2);

            point_t mp2{.x = poly2[midpoint2].x0, .y = poly2[midpoint2].y0};
            // a point of poly2 is inside poly1
            bool is_inside2 = is_inside_poly(poly1, mp2);
            if (is_inside2)
            { // midpoint of poly2 is inside poly1
                // pluck [b,c] - inclusive
                rpoly2 = simple_hacky_polygp_t{poly2t.begin() + b, poly2t.begin() + c + 1};  // inclusive (b) and (c)
            }
            else
            {
                // pluck ]a,d[ - inclusive
                rpoly2 = simple_hacky_polygp_t{poly2t.begin() + d, poly2t.end()}; // inclusive (d)
                // rpoly2.insert(poly2t.begin(), poly2t.begin() + a + 1); // inclusive (a)
                rpoly2.insert(/*to:*/ rpoly2.end(), /*from:*/ poly2t.begin(), poly2t.begin() + a + 1); // inclusive (a)
            }
        }
        if (mode == erasing_mod_t::no_erase)
        {
        }
        else
        {
            // note: assert(side_1[0] < side_1[1]);
            assert(new_point_indices1[0] < new_point_indices1[1]);
            assert(new_point_indices1[0] + 1 <= new_point_indices1[1] - 1); // becauwe we increased the second one
            if (mode == erasing_mod_t::erase_between)
            {
                side_index_int_t nexti = erase_between(rpoly, new_point_indices1);
                rpoly.insert( rpoly.begin() + nexti,  rpoly2.begin(),rpoly2.end() );
            }
            else if (mode == erasing_mod_t::erase_outside)
            {
                side_index_int_t nexti = erase_outwards(rpoly, new_point_indices1);
                rpoly.insert( rpoly.begin() + nexti,  rpoly2.begin(),rpoly2.end() );
            }
        }
        return rpoly;
    }
    // empty
    std::cout << "todo: I dont know if empty, or all of it" << std::endl;
    return simple_hacky_polygp_t{};
}

// , const pt2_t &point_
//     const point_t point{.x = point_.first, .y = point_.second};
bool is_inside_poly(const fixedsize_polygon_with_side_metadata_t &poly, const point_t &point)
{
    // https://stackoverflow.com/a/2922778/4374258  // no
    // https://math.stackexchange.com/a/4183060/236070

    if (build.debug)
    {
        std::cout << "which_side:";
    }

    constexpr double ε2 = 0.0000001;

    const auto xp = point.x;
    const auto yp = point.y;

    bool first = true;
    double last_sidedness = 1;
    // better if use circular_for()
    double x1 = (poly.end() - 1)->x0, y1 = (poly.end() - 1)->y0;
    for (const auto &p : poly)
    {
        double x2 = p.x0, y2 = p.y0;
        double which_side = (x2 - x1) * (yp - y1) - (xp - x1) * (y2 - y1);
        x1 = x2;
        y1 = y2;

        if (build.debug)
        {
            std::cout << which_side << ", ";
        }

        if (!first && last_sidedness * which_side < -ε2)
        {
            if (build.debug)
            {
                std::cout << "false." << std::endl;
            }

            return false;
        }
        last_sidedness = which_side;
        first = false;
    }
    if (build.debug)
    {
        std::cout << "true." << std::endl;
    }
    return true;
}

// followed by *.test.hpp
