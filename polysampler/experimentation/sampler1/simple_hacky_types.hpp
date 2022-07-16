#pragma once

// one of the many representaitons of point !
/*
template<typename real>
typedef std::pair<real,real> pt2_t_;
// a typedef cannot be a template
*/
typedef std::pair<double, double> pt2_t;

// use only for temporary (test, debug, etc), before conversions
// template<typename real>
typedef std::vector<pt2_t> simple_hacky_polygp_t;

