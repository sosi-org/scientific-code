export module export_svg_module;

// import point_t;

#include <iostream>

using std::string;

export void export_svg1();

void export_svg1() {
}
void export_svg2() {

   string s;
   s += R"XYZ(

    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

    <svg height="250" width="500"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink= "http://www.w3.org/1999/xlink"
    >
    <polygon points="220,10 300,210 170,250 123,234" style="fill:lime;stroke:purple;stroke-width:1" />
    Sorry, your browser does not support inline SVG.
    </svg>

   )XYZ";

   // See: https://en.cppreference.com/w/cpp/language/string_literal

   std::cout << s << std::endl;
}


int main (){
   export_svg1();
   return 0;
}