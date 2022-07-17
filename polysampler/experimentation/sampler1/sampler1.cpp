
/*
    not a well organised code,
    just code blocks that work.
    and to get an idea about the states, data transformations, caching states, etc.
*/

/*
#include <algorithm> // for reverse, unique

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

typedef model::d2::point_xy point;
*/

#include <vector>

// #include <string>

#include <cassert>

// #include <array>

// #include "../sampler1/point_t.hpp"
import point_t;

/*
(tesselation_t, points_t)
->
side_meta_data_t
array<side_meta_data_t> // has the circular thing
vector< array<side_meta_data_t> > // uneven size?
*/
//#include "../sampler1/side_meta_data_t.hpp"
//#include "side_meta_data_t.hpp"
import side_meta_data_t;
#include "line_intersection.hpp"

// #include "../sampler1/tesselation_t.hpp"
import tesselation_t;

// #include "./polyg.hpp"
import polyg;

// import svg_utils;
#include "svg_utils.hpp"

// convex
const tesselation_t &poly_poly_intersection(
    const tesselation_t &tesselation1,
    const tesselation_t &tesselation2,
    const points_t &points)
{
    return tesselation1;
}

// https://github.com/sosi-org/scientific-code/blob/main/beeseyes/pycode/polygon_sampler.py

// draft only:
// unordered_map
class vector_map
{
    // maps realtively sparse set of indices to another array of thinker data.
    /*
    .[row*M + col] -> int
    int -> vector[int]<my_struct>
    */
};

#include <iostream>
#include <functional>

// #include "side_meta_data_t.hpp"
import side_meta_data_t;

// #include "./polyg.hpp"
import polyg;

/*

    augmented sides.

    Keeps areference to coordinates,
    adds some metadata (augments) for each side.
    Each instance of this struct is for one instance of one side?

    no, it is a storage for all of them.

    Takes care of allocation, as well as, circular-loop

    Lifecycle:
        * empty but with allocated capacity
        * accumulating vector<>
        * fixedsize_polygon_with_side_metadata_t

*/
// almost like an accumulator, or rec_stat channel.
struct patch_t
{
    const points_t &coords_ref; // points_ref, rename -> points_coords_ref, coords_ref
    // const std::vector<side_point_t> &points_indices_ref;

    fixedsize_polygon_with_side_metadata_t side_meta_data; // for output

    /* accumulated sides: while being built */
    // std::vector<int>::size_type side_counter;
    // use side_meta_data.size() instead.

    /*
        Allocates placeholder. For speed.
    */
    // rename: points -> points_coords
    patch_t(const points_t &points, std::vector<int>::size_type nsides)
        : coords_ref(points) //, side_meta_data(nsides)
          ,
          side_meta_data()
    //, side_counter(0)
    {
        side_meta_data.reserve(nsides);
        // this->side_meta_data = side_meta_data_t(); //(0); // (nsides)
        std::cout << "polyg[" << nsides << "]: ";
        /* accumulated sides */
        // this->side_counter = 0;
    }

    /*
        augment and accumulate
    */
    // rename -> augment this with info from now that (another new side)
    // void augment_side(const int &from_idx, const int &to_idx /*, int idx*/)
    void augment_side(const int &from_idx, const int &to_idx /*, int idx*/)
    {
        const auto &p1 = this->coords_ref[from_idx];
        const auto &p2 = this->coords_ref[to_idx];
        std::cout << from_idx << ":" << (p1.tostr()) << "-" << to_idx << ":" << (p2.tostr()) << ". ";
        // index?
        /*
        side_meta_data[0].a = p2.x - p1.x;
        */

        // assert(this->side_counter <= side_meta_data.capacity()); // for performance only

        side_meta_data.push_back(side_meta_data_t{p1, p2});
        // side_meta_data[ this->side_counter ] = side_meta_data_t{p1, p2};

        // todo: clip away from the area/shape
    }

    // const fixedsize_polygon_with_side_metadata_t& finish()

    fixedsize_polygon_with_side_metadata_t finish()
    {
        std::cout << std::endl;
        // return side_meta_data; // move?

        // todo: store the value?
        // or maybe do something: area -> save in ...

        // compile-time size? I want const size, not compile-time size.
        // return std::array<side_meta_data_t,>(this->side_meta_data);

        // typedef std::unique_ptr<side_meta_data_t[]> fixedsize_polygon_with_side_metadata_t;

        // make_unique_for_overwrite
        // return std::make_unique<side_meta_data_t []> ( this->side_counter );

        // problem: how to copy?
        // return fixedsize_polygon_with_side_metadata_t(0);
        // return fixedsize_polygon_with_side_metadata_t(this->side_meta_data.begin(), this->side_meta_data.end());
        // return fixedsize_polygon_with_side_metadata_t(side_meta_data.size());

        // std::unique_ptr<side_meta_data_t []>
        // fixedsize_polygon_with_side_metadata_t fixedsize(side_meta_data.size());
        // std::unique_ptr<side_meta_data_t> fixedsize [side_meta_data.size()];
        /*
        std::generate(
            fixedsize.begin(), //std::begin(fixedsize),
            fixedsize.begin() + side_meta_data.size(), //std::end(fixedsize),
            []() { return std::make_unique<side_meta_data_t>(1); }
        );
        */
        /*
         fixedsize_polygon_with_side_metadata_t fixedsize(side_meta_data.size());
         std::copy(side_meta_data.begin(), side_meta_data.end(), fixedsize.get());
         return fixedsize;
         */
        // return fixedsize_polygon_with_side_metadata_t(0);
        return side_meta_data;
    }
};

// typedef std::vector<const side_point_t>::iterator side_it;

/*
    Goes through iterator range and applies the given lambda on consecutive pairs, circularly.
*/
template <typename IT>
void circular_for(IT _begin, IT _end, auto callback_pair)
{
    IT last_to;
    for (IT it = _begin; it < _end - 1; ++it)
    {
        const IT &next_it = std::next(it);
        callback_pair(it, next_it); // (from, to)
        last_to = next_it;
    }
    callback_pair(last_to, _begin);
}

/*
    clang++ sampler1.cpp -std=c++2b
*/

// todo: make return value optional at compile-time
// also pluggable multiple patch types: init, accum )iterate), finlise (closure).

// receive: points
// simple_polygi_t: polygon of integer vertices ("normalised")
fixedsize_polygon_with_side_metadata_t t2patch(const simple_polygi_t &polyg, const points_t &points)
{

    patch_t patch{points, polyg.size()};

    // lambda capture list:  [patch_t&patch]  -> [&patch]
    auto callback_pair = [&patch]<typename IT>(const IT &from_it, const IT &to_it)
    {
        // todo: move to a seaparate fixedsize_polygon_with_side_metadata_t element
        //       as opposed to storing it inside the `patch` itself.
        patch.augment_side(*from_it, *to_it);
        // cout << *from_i << ',' << *next_it <<' ';
    };
    // circular_for_pairs
    circular_for(polyg.begin(), polyg.end(), callback_pair);

    std::cout << std::endl;

    // patch.finish();
    return patch.finish();
}
/*
template <typename real>
side_side_intersection_solution_t<real>
intersect_lines(const side_meta_data_t &side1, const side_meta_data_t &side2);
*/

// std::function<void (const patch_t&, const side_it&, const side_it&)> augment_side

// sepaate function for each?
void augment_tesselation_polygons(const tesselation_t &trigulation, const points_t &points)
{
    // fixedsize_polygon_with_side_metadata_t *x0;
    std::vector<fixedsize_polygon_with_side_metadata_t> r;
    /* -> fixedsize_polygon_with_side_metadata_t*/
    traverse_tesselation(
        trigulation, points, [&points](const auto &polyg)
        { return t2patch(polyg, points); },
        r);
    // now r contains the augmented data structure:
    // debug print:
    for (const auto &polysides : r)
    {
        for (const side_meta_data_t &side : polysides)
        {
            std::cout << "( o:" << side.x0 << "," << side.y0 << "; d:" << side.dx << "," << side.dy << ") ";
        }
        std::cout << std::endl;
        std::cout << "intersections: ";
        // double-circular_for !
        circular_for(polysides.begin(), polysides.end(), [](auto side1, auto side2)
                     {
                        // todo: remove explicit concrete type double
            auto ip = intersect_lines_segment<double>(*side1, *side2);
            std::cout << "(" << ip.x << "," << ip.y << ") "; });
        std::cout << std::endl;
        /*
        auto ip = intersect_lines<double>(polysides[0], polysides[1]);
        std::cout << "\nintersection: " << ip.x << "," << ip.y << std::endl;
        */
    }
}

// Good test scenario:
// (x1,y1), (x2,y2) , (x3,y3) -> should give x2,y2

// todo: move to side_meta_data_t.hpp

void intersect_polys(fixedsize_polygon_with_side_metadata_t poly1, fixedsize_polygon_with_side_metadata_t poly2)
{
    for (const side_meta_data_t &side_ : poly1)
    {
        // const side_meta_data_t  & side_ = side;
        std::cout << "( o:" << side_.x0 << "," << side_.y0 << "d:" << side_.dx << "," << side_.dy << ") ";
    }
}
/*
#include <random>
std::random_device rdev;
std::mt19937 rngmt(rdev());
*/
std::uniform_real_distribution<double> dist(-1 - 2, 2 + 2); // distribution in range [-1.0, 2.0]
// from https://stackoverflow.com/a/13445752/4374258
// https://stackoverflow.com/a/19666713/4374258

// const int SAMPLING_FACTOR = 16;
// const int SAMPLING_FACTOR = 4 * 4 * 4;
const int SAMPLING_FACTOR = 4 * 4 * 4 * 4 * 4;

template <typename real>
// std::vector<point_t>
auto generate_helper_annots(const tesselation_t &trigulation, const points_t &vertex_coords)
{

    std::vector<point_t> helper_points{};
    std::vector<std::vector<point_t>> helper_points1{};
    std::vector<std::vector<point_t>> helper_points2d{};

    std::vector<side_meta_data_t> helper_lines{};

    traverse_tesselation(
        trigulation, vertex_coords, [vertex_coords, &helper_points, &helper_lines](const auto &polyg)
        {
            // per face

            // string point_seq{};

            auto callback =  [vertex_coords, &helper_points, &helper_lines](auto from_vert_it, auto to_vert_it)
            //-> std::vector<point_t>
            {

                const point_t & point1 = vertex_coords[*from_vert_it];
                const point_t & point2 = vertex_coords[*to_vert_it];
                side_meta_data_t md = side_meta_data_t{point1, point2};

                for(int i = 0; i < SAMPLING_FACTOR; ++i) {
                //point_t p1{0.5, 0.5}, p2{2, 2};
                point_t p1{dist(rngmt), dist(rngmt)},
                        p2{dist(rngmt), dist(rngmt)};
                side_meta_data_t s0md = side_meta_data_t{p1, p2};

                auto ip = intersect_lines_segment<real>(s0md, md);
                //std::cout << "(" << ip.x << "," << ip.y << ") ";
                if (ip.intersect) {
                   helper_points.push_back(point_t{ip.x,ip.y});
                    // helper_lines.push_back(s0md);
                }
                }
                //return std::vector<point_t>{};
            };

            circular_for(polyg.begin(), polyg.end(),callback);
            return std::vector<point_t>{}; },
        helper_points2d);

    /*
        return helper_points;
    },
            helper_points2d
        );
        */
    /*
    //intersect_lines_segment(s0, const side_meta_data_t &side2);

    return helper_points;
    }
    */
    return std::make_pair(helper_points, helper_lines);
}

#include "./polygon_area.hpp"
#include "./polygon_area.test.hpp"

#include "./cpolyg_intersection.hpp"
#include "./cpolyg_intersection.test.hpp"

void run_tests()
{
    test1_convex_polygon_area();
    test2_convex_polygon_area();
    test1_cpoly_intersection__two_points();
    test2_cpoly_intersection__two_points();
    test1_cpoly_intersection__produced();
    test1_insideness();
}

int main()
{
    run_tests();

    std::cout << "-------------------------" << std::endl;

    /*
    points_t points = // {{1,2}, {3,4}};
        {point_t{0, 1}, {0.4, 0.8}, {0.8, 0.5}, {1, 0.3}, {0, -1}, {-1, 0}, {-0.3, 0.3}};

    tesselation_t trigulation = {{1, 2, 3, 4, 5}, {1, 2, 6}};
    */

    full_tesselation example2{
        // points:
        points_t{
            point_t{-0.2, 0.8}, // 0
            {0.4, 0.8},         // 1
            {0.9, 0.5},         // 2
            {1, 0.0},           // 3
            {0, -1},            // 4
            {-1, 0},            // 5
            {-0.1, -0.2},       // 6
        },
        // trigulation:
        tesselation_t{{
            //{1, 2, 3, 4, 5},
            {1, 2, 6},
            //{0,1,2,6,5},
            {0, 1, 6, 5},
            {2, 3, 4, 6},
            {4, 6, 5},
        }}

    };

    // bool svg_only = (argc > 0);

    augment_tesselation_polygons(example2.trigulation, example2.points);

    // std::vector<point_t> helper_points
    // std::pair<std::vector<point_t> , std::vector<intersect_lines_segment<real> > > =
    auto [hpoints, hlines] =
        generate_helper_annots<double>(example2.trigulation, example2.points); // for debugging

    save_svg_file("./output.svg", example2.trigulation, example2.points, hpoints, hlines, svg_utils<double>::svgctx_t{});

    return 0;
}
// ExecutionPolicy
