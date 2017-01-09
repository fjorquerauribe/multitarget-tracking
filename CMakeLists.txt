cmake_minimum_required(VERSION 2.8)
project( mtt )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")

#SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11 -fopenmp")
SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11")

SET (CMAKE_CXX_FLAGS_DEBUG          "-g -traceback")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "/usr/local/include/eigen3/" )
include_directories( "src/utils/" )

add_executable( mtt src/mtt.cpp src/utils/image_generator.cpp src/detectors/hog_detector.cpp src/dpp.cpp src/utils/cnn_reader.cpp)
target_link_libraries( mtt ${OpenCV_LIBS} -lpthread)
