
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

#include "sampler1/point_t.hpp"

/*
(triaglation_t, points_t)
->
side_meta_data_t
array<side_meta_data_t> // has the circular thing
vector< array<side_meta_data_t> > // uneven size?
*/
#include "sampler1/side_meta_data_t.hpp"

void poly_poly_intersection(const triaglation_t &triaglation, const points_t &points)
{
    ;
}


#include "sampler1/tesselation_t.hpp"

typedef std::vector<point_t> points_t;
// Delaunay triangulation

points_t points = // {{1,2}, {3,4}};
    {point_t{0, 1}, {0.4, 0.8}, {0.8, 0.5}, {1, 0.3}, {0, -1}, {-1, 0}, {-0.3, 0.3}};

triaglation_t trigulation = {{1, 2, 3, 4, 5}, {1, 2, 6}};

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

// almost like an accumulator, or rec_stat channel.
struct patch_t
{
    const points_t &points_ref;
    std::vector<side_meta_data_t> side_meta_data; // for output

    patch_t(std::vector<int>::size_type nsides, const points_t &points)
        : points_ref(points), side_meta_data(nsides)
    {
        std::cout << nsides << ": ";
    }
    void do_side(const int &from_idx, const int &to_idx /*, int idx*/)
    {
        const auto &p1 = this->points_ref[from_idx];
        const auto &p2 = this->points_ref[to_idx];
        std::cout << from_idx << ":" << (p1.tostr()) << "-" << to_idx << ":" << (p2.tostr()) << ".";
        // index?
        /*
        side_meta_data[0].a = p2.x - p1.x;
        */
        side_meta_data[0] = side_meta_data_t(p1, p2);

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

void traverse(const triaglation_t &trigulation, const points_t &points)
{
    for (auto plg = trigulation.begin(); plg < trigulation.end(); ++plg)
    {
        const auto &polyg_i = *plg;

        patch_t patch{polyg_i.size(), points};

        for (auto vert = polyg_i.begin(); vert < polyg_i.end() - 1; ++vert)
        {
            const auto &from_i = *vert;
            const auto &to_i = *(vert + 1);
            patch.do_side(from_i, to_i);
        }
        patch.do_side(*(polyg_i.end() - 1), *(polyg_i.begin()));

        patch.finish();
        // std::vector<side_meta_data_t> q = patch.finish();
    }
}
/*
    clang++ sampler1.cpp -std=c++2b
*/

int main()
{
    std::cout << "hi" << std::endl;

    traverse(trigulation, points);
    return 0;
}
// ExecutionPolicy
