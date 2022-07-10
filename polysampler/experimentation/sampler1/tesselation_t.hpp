export module tesselation_t;
// #pragma once

#include <vector>

// #include "point_t.hpp"
import point_t;

export
{
    // typedef struct {struct{int point_idx} main; struct {double a;} cache;} side_point;
    typedef int side_point_t; // FROM, and TO is the next one.

    typedef std::vector<std::vector<side_point_t>> tesselation_t; // old name: triaglation_t
    // Also add meta-data to each item.

    typedef std::vector<point_t> points_t;
    // Delaunay triangulation
}
/*
todo: rename `points_t`. Too similar to `point_t`

*/

export template <typename func, typename resultt>
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
