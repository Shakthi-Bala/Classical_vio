cmake_minimum_required(VERSION 3.16)
project(msckf_vio_cpp LANGUAGES C CXX)   # C: Ceres -> SuiteSparse looks for OpenMP_C

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # lets the editor find headers

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)         # -O3; never time a Debug build
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video features2d calib3d)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED CONFIG COMPONENTS headers)   # boost::math (header-only)
find_package(Ceres 2.1 REQUIRED)                         # sliding-window back end

# The VIO itself, as a library so the program and the tests share it.
add_library(msckf
  src/math_utils.cpp
  src/config.cpp
  src/dataset.cpp
  src/feature.cpp
  src/image_processor.cpp
  src/msckf.cpp
  src/swo/imu_preintegration.cpp
  src/swo/marginalization.cpp
  src/swo/sliding_window.cpp
)
target_include_directories(msckf PUBLIC include)
target_link_libraries(msckf PUBLIC
  Eigen3::Eigen
  ${OpenCV_LIBS}
  yaml-cpp::yaml-cpp
  Boost::headers
  Ceres::ceres
)
target_compile_options(msckf PRIVATE -Wall -Wextra -Wpedantic)

add_executable(main_euroc src/main_euroc.cpp)
target_link_libraries(main_euroc PRIVATE msckf)
target_compile_options(main_euroc PRIVATE -Wall -Wextra -Wpedantic)

# Unit tests (gtest). Run with: ctest --test-dir build --output-on-failure
enable_testing()
find_package(GTest REQUIRED)
add_executable(msckf_tests
  test/test_math_utils.cpp
  test/test_feature.cpp
  test/test_swo.cpp
)
target_link_libraries(msckf_tests PRIVATE msckf GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(msckf_tests)
