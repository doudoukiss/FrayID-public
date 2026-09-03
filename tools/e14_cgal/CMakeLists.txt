cmake_minimum_required(VERSION 3.22)
project(frayid_e14_cgal LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

find_package(CGAL 6.2 EXACT REQUIRED COMPONENTS Core)

add_executable(frayid_e14_dyadic_envelope dyadic_envelope.cpp)
target_compile_features(frayid_e14_dyadic_envelope PRIVATE cxx_std_20)
target_link_libraries(frayid_e14_dyadic_envelope PRIVATE CGAL::CGAL CGAL::CGAL_Core)
target_compile_options(frayid_e14_dyadic_envelope PRIVATE -Wall -Wextra -Wpedantic)
