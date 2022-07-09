
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

typedef std::vector<side_point_t>  vertiex_indices_t; // vertices; // without coords, just int, refering to the coords index

//const std::vector<side_point_t> &points_indices_ref;  = vertices
// const vertiex_indices_t &vertiex_indices,

/*
    Keeps areference to coordinates,
    adds some metadata (augments) for each side.
    Each instance of this struct is for one instance of one side?

    no, it is a storage for all of them.

    Takes care of allocation, as well as, circular-loop
*/
// almost like an accumulator, or rec_stat channel.
struct patch_t
{
    const points_t &points_ref; // rename -> points_coords_ref
    // const std::vector<side_point_t> &points_indices_ref;

    std::vector<side_meta_data_t> side_meta_data; // for output

    /*
        Allocates placeholder. For speed.
    */
    //rename: points -> points_coords
    patch_t(std::vector<int>::size_type nsides, const points_t &points)
        : points_ref(points) //, side_meta_data(nsides)
          ,
          side_meta_data()
    {
        side_meta_data.reserve(nsides);
        // this->side_meta_data = side_meta_data_t(); //(0); // (nsides)
        std::cout << nsides << ": ";
    }

    // rename -> augment this with info from now that (another new side)
    //void do_side(const int &from_idx, const int &to_idx /*, int idx*/)
    void do_side(const int &from_idx, const int &to_idx /*, int idx*/)
    {
        const auto &p1 = this->points_ref[from_idx];
        const auto &p2 = this->points_ref[to_idx];
        std::cout << from_idx << ":" << (p1.tostr()) << "-" << to_idx << ":" << (p2.tostr()) << ". ";
        // index?
        /*
        side_meta_data[0].a = p2.x - p1.x;
        */
        auto x = side_meta_data_t{p1, p2};
        side_meta_data[0] = x;

        // todo: clip away from the area/shape
    }

    // const std::vector<side_meta_data_t>& finish()
    void finish()
    {
        std::cout << std::endl;
        // return side_meta_data; // move?

        // todo: store the value?
        // or maybe do something: area -> save in ...
    }
};

//typedef std::vector<const side_point_t>::iterator side_it;

/*
    Goes through iterator range and applies the given lambda on consecutive pairs, circularly.
*/
void circular_for(auto _begin, auto _end, auto callback_pair) {

    auto first = _begin;
    auto last_to = _begin;
    for (auto it = _begin; it < _end - 1; ++it)
    {
        const auto &from_i = *it;
        const auto &to_i = *(std::next(it)); // next
        // patch.do_side(from_i, to_i);
        callback_pair(&from_i, &to_i);
        // do_side(patch, from_i, to_i);
        // last = it; // rename: it -> vert_p
        // last_to = to_i;
        last_to = std::next(it);
    }
    //patch.do_side(*(polyg_i.end() - 1), *(polyg_i.begin()));
    // patch.do_side(*last_to, *first);
    // do_side(patch, last_to, first);
    callback_pair(last_to, first);
}

// std::function<void (const patch_t&, const side_it&, const side_it&)> do_side

void traverse(const tesselation_t &trigulation, const points_t &points /*, auto do_side*/)
{
    for (auto plg = trigulation.begin(); plg < trigulation.end(); ++plg)
    {
        const auto &polyg_i = *plg;

        patch_t patch{polyg_i.size(), points};

        // lambda capture list:  [patch_t&patch]  -> [&patch]
        auto callback_pair = [&patch]<typename IT>(const IT &from_it, const IT &to_it) {
            patch.do_side(*from_it, *to_it);
            // cout << *from_i << ',' << *to_i <<' ';
        };
        // circular_for_pairs
        circular_for(polyg_i.begin(), polyg_i.end(), callback_pair);

        std::cout << std::endl;

        patch.finish();
        // std::vector<side_meta_data_t> q = patch.finish();
    }
}
/*
    clang++ sampler1.cpp -std=c++2b
*/




string export_svg3(double xi[4][2]) {


   string point_seq{};
   for(int i = 0; i < 4; ++i ) {
       // point_seq = point_seq + std::format("{} ", (int) (xi[i]));
       point_seq = point_seq + " " + std::to_string(xi[i][0]) + "," + std::to_string(xi[i][1]);
   }
   string s{};
   s += R"XYZ(

    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

    <svg height="250" width="500"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink= "http://www.w3.org/1999/xlink"
    >
    <polygon points="$$POINTS$$" style="fill:white;stroke:blue;stroke-width:2" />
    Sorry, your browser does not support inline SVG.
    </svg>

   )XYZ";

   // std::string::replace
   s = std::regex_replace(s, std::regex("\\$\\$POINTS\\$\\$"), point_seq);
   return s;
}
int main()
{
    // std::cout << "hi" << std::endl;

    double xi[4][2] = {{220,0}, {300,50}, {170,70}, {0,100}};
    std::cout << export_svg3(xi) << std::endl;

    //std::cout << "ok" << std::endl;

    traverse(trigulation, points/*, callback*/);

    return 0;
}
// ExecutionPolicy
