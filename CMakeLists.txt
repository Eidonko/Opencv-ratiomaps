cmake_minimum_required(VERSION 2.8)
project( RatioMap )
find_package( OpenCV REQUIRED )
add_executable( ratiomap ratiomap.cpp )
target_link_libraries( ratiomap ${OpenCV_LIBS} )

