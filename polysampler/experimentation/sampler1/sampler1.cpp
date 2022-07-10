
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

// std::function<void (const patch_t&, const side_it&, const side_it&)> augment_side

template <typename func, typename resultt>
// template <typename func>
// template <typename resultt>
// std::vector<decltype( process_polyg_callback() )>
// std::vector<resultt>
void traverse_tesselation(const tesselation_t &trigulation, const points_t &points, func process_polyg_callback
                          // resultt*
                          ,
                          std::vector<resultt> &accum
                          /*, auto augment_side*/)
{
    // std::vector<resultt> accum{0};

    for (auto plg_it = trigulation.begin(); plg_it < trigulation.end(); ++plg_it)
    {
        // Take each polygon from the tesselation
        const auto &polyg = *plg_it;

        // fixedsize_side_metadata_t r;
        resultt r = process_polyg_callback(polyg);
        // process_polyg_callback(polyg);
        //  if you want to keep them:
        //  std::vector<side_meta_data_t> q = patch.finish();
        accum.push_back(r);
    }
    // return accum;
}
/*
    clang++ sampler1.cpp -std=c++2b
*/

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
}

string export_svg3(const tesselation_t &trigulation, const points_t &vertex_coords)
{

    string *x0;

    // todo: const vertex_coords

    // string total_point_seq;
    // std::vector<string> total_point_seq;
    std::vector<string> total_point_seq;
    // /* -> string */
    traverse_tesselation(
        trigulation, vertex_coords, [vertex_coords, &total_point_seq](const auto &polyg)
        {
        // per face

        string point_seq{};

        circular_for(polyg.begin(), polyg.end(), [&point_seq, vertex_coords]<typename IT>(const IT &from_vert, const IT &to_vert)  {
            // per edge
            // point_seq = point_seq + std::format("{} ", (int) (xi[i]));
            // point_seq = point_seq + " " + std::to_string(xi[i][0]) + "," + std::to_string(xi[i][1]);
            //std::cout << from_vert;
            const point_t & point = vertex_coords[*from_vert];
            double x = point.x * 150 + 200;
            double y = point.y * 150 + 200;
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
    ts = std::regex_replace(ts, std::regex("\\$WIDTH"), "500");
    ts = std::regex_replace(ts, std::regex("\\$HEIGHT"), "400");
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
