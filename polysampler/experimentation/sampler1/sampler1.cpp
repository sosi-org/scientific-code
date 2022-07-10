
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

#include <string>

#include <format>
#include <regex>
#include <cassert>

// #include <array>

using std::string;
// using std::format;

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
#include "side_meta_data_t.hpp"

// #include "../sampler1/tesselation_t.hpp"
import tesselation_t;

// convex
const tesselation_t &poly_poly_intersection(
    const tesselation_t &tesselation1,
    const tesselation_t &tesselation2,
    const points_t &points)
{
    return tesselation1;
}

points_t points = // {{1,2}, {3,4}};
    {point_t{0, 1}, {0.4, 0.8}, {0.8, 0.5}, {1, 0.3}, {0, -1}, {-1, 0}, {-0.3, 0.3}};

tesselation_t trigulation = {{1, 2, 3, 4, 5}, {1, 2, 6}};

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

#include "side_meta_data_t.hpp"

typedef std::vector<side_point_t> vertiex_indices_t; // vertices; // without coords, just int, refering to the coords index

// const std::vector<side_point_t> &points_indices_ref;  = vertices
//  const vertiex_indices_t &vertiex_indices,

// typedef std::unique_ptr<side_meta_data_t[]> fixedsize_side_metadata_t;
typedef std::vector<side_meta_data_t> fixedsize_side_metadata_t;

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
        * fixedsize_side_metadata_t

*/
// almost like an accumulator, or rec_stat channel.
struct patch_t
{
    const points_t &coords_ref; // points_ref, rename -> points_coords_ref, coords_ref
    // const std::vector<side_point_t> &points_indices_ref;

    std::vector<side_meta_data_t> side_meta_data; // for output

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

    // const std::vector<side_meta_data_t>& finish()

    fixedsize_side_metadata_t finish()
    {
        std::cout << std::endl;
        // return side_meta_data; // move?

        // todo: store the value?
        // or maybe do something: area -> save in ...

        // compile-time size? I want const size, not compile-time size.
        // return std::array<side_meta_data_t,>(this->side_meta_data);

        // typedef std::unique_ptr<side_meta_data_t[]> fixedsize_side_metadata_t;

        // make_unique_for_overwrite
        // return std::make_unique<side_meta_data_t []> ( this->side_counter );

        // problem: how to copy?
        // return fixedsize_side_metadata_t(0);
        // return fixedsize_side_metadata_t(this->side_meta_data.begin(), this->side_meta_data.end());
        // return fixedsize_side_metadata_t(side_meta_data.size());

        // std::unique_ptr<side_meta_data_t []>
        // fixedsize_side_metadata_t fixedsize(side_meta_data.size());
        // std::unique_ptr<side_meta_data_t> fixedsize [side_meta_data.size()];
        /*
        std::generate(
            fixedsize.begin(), //std::begin(fixedsize),
            fixedsize.begin() + side_meta_data.size(), //std::end(fixedsize),
            []() { return std::make_unique<side_meta_data_t>(1); }
        );
        */
        /*
         fixedsize_side_metadata_t fixedsize(side_meta_data.size());
         std::copy(side_meta_data.begin(), side_meta_data.end(), fixedsize.get());
         return fixedsize;
         */
        // return fixedsize_side_metadata_t(0);
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


template <typename real>
void intersect_lines(const side_meta_data_t &side1, const side_meta_data_t &side2);

// std::function<void (const patch_t&, const side_it&, const side_it&)> augment_side

void augment_tesselation_polygons(const tesselation_t &trigulation, const points_t &points)
{
    // fixedsize_side_metadata_t *x0;
    std::vector<fixedsize_side_metadata_t> r;
    /* -> fixedsize_side_metadata_t*/
    traverse_tesselation(
        trigulation, points, [&points](const auto &polyg)
        {

        patch_t patch{points, polyg.size()};

        // lambda capture list:  [patch_t&patch]  -> [&patch]
        auto callback_pair = [&patch]<typename IT>(const IT &from_it, const IT &to_it) {
            patch.augment_side(*from_it, *to_it);
            // cout << *from_i << ',' << *next_it <<' ';
        };
        // circular_for_pairs
        circular_for(polyg.begin(), polyg.end(), callback_pair);

        std::cout << std::endl;

        //patch.finish();
        return patch.finish(); },
        r);
    // now r contains the augmented data structure:
    // debug print:
    for(const auto& poly : r) {
        //fixedsize_side_metadata_t poly;
        for (const side_meta_data_t& side_ : poly) {
            // const side_meta_data_t  & side_ = side;
            std::cout << "( o:" << side_.x0 << ","<< side_.y0 << "d:" <<  side_.dx <<","<< side_.dy << ") ";
        }
        intersect_lines<double>(poly[0], poly[1]);
    }
}

template <typename real>
struct side_side_intersection_solution_t
{
    bool intersect;
    // real condition_number; // ie do not intersect in case of parallel. or too far.
    real x;
    real y;
};

template <typename real>
inline side_side_intersection_solution_t<real> solveAxb(real a, real b, real c, real d, real x, real y)
{
    real det = (a * d - b * c);
    // real [d,-b, -c, a] / det;
    //[d,-b, -c, a] * [x,y].T / det
    //[d,-b,  -c, a] * [x,y].T / det
    //[d *x  + -b*y,  -c*x+ a*y] / det
    real x1 = (+d * x - b * y) / det;
    real y1 = (-c * x + a * y) / det;

    return side_side_intersection_solution_t<real>{true, x1, y1};
}
// Good test scenario:
// (x1,y1), (x2,y2) , (x3,y3) -> should give x2,y2

/*
template <typename real>
inline inv2x2( real a,  real b,  real c, real d) {
    real det = (a*d-b*c);
    real (d,-b, -c, a) / det;
    (d,-b, -c, a) / det * B
}
*/

// todo: move to side_meta_data_t.hpp
template <typename real>
// bool
void intersect_lines(const side_meta_data_t &side1, const side_meta_data_t &side2)
{
    /*
    double x0, y0;
    double dx, dy;
    double x = ...;

    // a1 * x + b1 * y = c1
    // a2 * x + b2 * y = c2
    [a1, b1]
    [a2, b2]
    [c1,c2].T
    det = (a1);
    */

    real dx1 = side1.dx;
    real dy1 = side1.dy;
    real dx2 = side2.dx;
    real dy2 = side2.dy;
    real x10 = side1.x0;
    real y10 = side1.y0;
    real x20 = side2.x0;
    real y20 = side2.y0;
    /*
    line equation: (1/dy|-1/dx) . ( (x|y) - (x0|y0) ) == 0
    a x + b y == c
         a = 1/dy
         b = -1/dx
         c = a*x0 + b*y0
    */
    double a1 = 1 / dy1, b1 = -1 / dx1;
    double a2 = 1 / dy2, b2 = -1 / dx2;
    // returntype<real> funcname<real>  --> why none is needed?
    side_side_intersection_solution_t ip =
        solveAxb(
            a1, b1,
            a2, b2,

            a1 * x10 + b1 * y10, a2 * x20 + b2 * y20);

    std::cout << "intersection: " << ip.x << "," << ip.y << std::endl;
}

void intersect_polys(fixedsize_side_metadata_t poly1, fixedsize_side_metadata_t poly2)
{
    for (const side_meta_data_t &side_ : poly1)
    {
        // const side_meta_data_t  & side_ = side;
        std::cout << "( o:" << side_.x0 << "," << side_.y0 << "d:" << side_.dx << "," << side_.dy << ") ";
    }
}

string export_svg3(const tesselation_t &trigulation, const points_t &vertex_coords)
{
    /* svg parameters */
    const struct
    {
        double scale = 150, offsetx = 200, offsety = 200;
        string width = "500", height = "400";
    } svgctx;

    // todo: const vertex_coords

    // string total_point_seq;
    // std::vector<string> total_point_seq;
    std::vector<string> total_point_seq;
    // /* -> string */
    traverse_tesselation(
        trigulation, vertex_coords, [vertex_coords, &total_point_seq, svgctx](const auto &polyg)
        {
        // per face

        string point_seq{};

        circular_for(polyg.begin(), polyg.end(), [&point_seq, vertex_coords, svgctx]<typename IT>(const IT &from_vert, const IT &to_vert)  {
            // per edge
            // point_seq = point_seq + std::format("{} ", (int) (xi[i]));
            // point_seq = point_seq + " " + std::to_string(xi[i][0]) + "," + std::to_string(xi[i][1]);
            //std::cout << from_vert;
            const point_t & point = vertex_coords[*from_vert];
            double x = point.x * svgctx.scale + svgctx.offsetx;
            double y = point.y * svgctx.scale + svgctx.offsety;
            point_seq = point_seq + " " + std::to_string(x) + "," + std::to_string(y);
            std::cout << " " + std::to_string(x) + "," + std::to_string(y);
        });

        std::cout << std::endl;

        //total_point_seq = point_seq;

        return point_seq; },
        total_point_seq);

    string polygon_template = R"XYZ(
        <polygon points="$$POINTS$$" style="fill:yellow;stroke:blue;stroke-width:4" />
    )XYZ";
    string polyss{};
    for (const auto &point_seq_str : total_point_seq)
    {
        polyss += std::regex_replace(polygon_template, std::regex("\\$\\$POINTS\\$\\$"), point_seq_str);
    }

    const string svg_template = R"XYZ(
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <svg height="$HEIGHT" width="$WIDTH"
        xmlns="http://www.w3.org/2000/svg" xmlns:xlink= "http://www.w3.org/1999/xlink"
        >

        $$POLYG$$
        Sorry, your browser does not support inline SVG.
    </svg>
   )XYZ";

    // rename: total
    string ts{};
    // std::string::replace
    ts = std::regex_replace(svg_template, std::regex("\\$\\$POLYG\\$\\$"), polyss);
    ts = std::regex_replace(ts, std::regex("\\$WIDTH"), svgctx.width);
    ts = std::regex_replace(ts, std::regex("\\$HEIGHT"), svgctx.height);
    return ts;
}

int main()
{
    // std::cout << "hi" << std::endl;

    std::cout << std::endl
              << std::endl;
    // double xi[4][2] = {{220,0}, {300,50}, {170,70}, {0,100}};
    // std::cout << export_svg3(xi) << std::endl;
    std::cout << export_svg3(trigulation, points) << std::endl;
    std::cout << std::endl
              << std::endl;

    // std::cout << "ok" << std::endl;

    augment_tesselation_polygons(trigulation, points /*, callback*/);

    return 0;
}
// ExecutionPolicy
