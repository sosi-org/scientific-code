#pragma once

// export module svg_utils;

import point_t;
import tesselation_t;
//todo:
// #include "side_meta_data_t.hpp"
import side_meta_data_t;

// import std.fstream;
// import <std.fstream>;

#include <string>
using std::string;
// todo: std::string -> string
/*
#include <format>
// using std::format;
*/
#include <regex>
#include <iostream>
#include <fstream>
// todo:
// import std.iostream;
// import std.fstream;
// import <std.fstream>;


//export
template <typename real>
class svg_utils
{
public:
    /* svg parameters */
    struct svgctx_t
    {
        real scale = 1.0, offsetx = 0.0, offsety = 0.0;
        string width = "5cm", height = "5cm";
        string view_box = "-1 -1 4 4";
        string stroke_width = "0.04";
    };

    // static std::string multi_polygon(total_point_seq)
    static std::string tessellation(const svgctx_t &svgctx, const tesselation_t &trigulation, const points_t &vertex_coords)
    {

        // todo: const vertex_coords

        // string total_point_seq;
        // std::vector<string> total_point_seq;
        std::vector<string> total_point_seq;
        // /* -> string */
        traverse_tesselation(
            trigulation, vertex_coords, [vertex_coords, &total_point_seq, svgctx](const auto &polyg)
            {

                //todo: indent
        // per face

        string point_seq{};

        circular_for(polyg.begin(), polyg.end(), [&point_seq, vertex_coords, svgctx]<typename IT>(const IT &from_vert, const IT &to_vert)  {
            // per edge
            // point_seq = point_seq + std::format("{} ", (int) (xi[i]));
            // point_seq = point_seq + " " + std::to_string(xi[i][0]) + "," + std::to_string(xi[i][1]);
            //std::cout << from_vert;
            const point_t & point = vertex_coords[*from_vert];
            real x = point.x * svgctx.scale + svgctx.offsetx;
            real y = point.y * svgctx.scale + svgctx.offsety;
            point_seq = point_seq + " " + std::to_string(x) + "," + std::to_string(y);
            std::cout << " " + std::to_string(x) + "," + std::to_string(y);
        });

        std::cout << std::endl;

        //total_point_seq = point_seq;

        return point_seq; },
            total_point_seq);

        const string polygon_template = R"XYZ(
            <polygon points="$$POINTS$$" style="fill:yellow;stroke:blue;stroke-width:$STROKE_WIDTH" opacity="1" />
        )XYZ";
        string polyss{};
        for (const auto &point_seq_str : total_point_seq)
        {
            polyss += std::regex_replace(polygon_template, std::regex("\\$\\$POINTS\\$\\$"), point_seq_str);
        }
        return polyss;
    }

    static std::string helper_dot(real x, real y)
    {
        const string helperdot_template = R"XYZ(
            <circle cx="$X" cy="$Y" r="0.05" fill="red" />
        )XYZ";

        string s = helperdot_template;
        s = std::regex_replace(s, std::regex("\\$X"), std::to_string(x));
        s = std::regex_replace(s, std::regex("\\$Y"), std::to_string(y));
        return s;
    }
    static std::string helper_line(side_meta_data_t line_seg)
    {
        const string helperline_template = R"XYZ(
            <polyline points=" $X1,$Y1 $X2,$Y2" style="stroke:grey;stroke-width:0.01" />
        )XYZ";

        string s = helperline_template;
        s = std::regex_replace(s, std::regex("\\$X1"), std::to_string(line_seg.x0));
        s = std::regex_replace(s, std::regex("\\$Y1"), std::to_string(line_seg.y0));
        s = std::regex_replace(s, std::regex("\\$X2"), std::to_string(line_seg.x1));
        s = std::regex_replace(s, std::regex("\\$Y2"), std::to_string(line_seg.y1));
        return s;
    }

    static string generate_svg(const tesselation_t &trigulation, const points_t &vertex_coords, const std::vector<point_t> extra_points, const std::vector<side_meta_data_t> &helper_lines)
    {

        const svg_utils::svgctx_t svgctx;

        const string polyss = tessellation(svgctx, trigulation, vertex_coords);

        string helper_points = "";
        for (point_t p : extra_points)
        {
            helper_points += helper_dot(p.x, p.y);
        }

        string helper_lines_s = "";
        for (auto hl : helper_lines)
        {
            helper_lines_s += helper_line(hl);
        }

        const string svg_template = R"XYZ(
        <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
            <svg height="$HEIGHT" width="$WIDTH"
                xmlns="http://www.w3.org/2000/svg" xmlns:xlink= "http://www.w3.org/1999/xlink"
                viewBox="$VIEWBOX"
                >

                $$POLYG$$

                $$HELPER_DOTS$$

                Sorry, your browser does not support inline SVG.
            </svg>
        )XYZ";

        // rename: total
        string ts = svg_template;
        // std::string::replace

        ts = std::regex_replace(ts, std::regex("\\$\\$POLYG\\$\\$"), polyss);
        ts = std::regex_replace(ts, std::regex("\\$\\$HELPER_DOTS\\$\\$"), helper_points + "\n" + helper_lines_s);

        ts = std::regex_replace(ts, std::regex("\\$WIDTH"), svgctx.width);
        ts = std::regex_replace(ts, std::regex("\\$HEIGHT"), svgctx.height);
        ts = std::regex_replace(ts, std::regex("\\$VIEWBOX"), svgctx.view_box);
        ts = std::regex_replace(ts, std::regex("\\$STROKE_WIDTH"), svgctx.stroke_width);
        return ts;
    }
};

//export
void save_svg_file(const string &file_name, const auto &trigulation, const auto &points,
                   const std::vector<point_t> &helper_points,
                   const std::vector<side_meta_data_t> &helper_lines)
{
    std::wofstream file;
    file.open(file_name.c_str());

    // will it be efficient?
    string contents = svg_utils<double>::generate_svg(trigulation, points, helper_points, helper_lines);
    std::cout << contents << std::endl;

    file << contents.c_str();

    file.close();
}
