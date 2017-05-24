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
#include_directories( "/usr/include/eigen3/" )
include_directories( "/usr/include/eigen3/" "/usr/local/include/eigen3/" )
include_directories( "src/utils/" )
include_directories( "src/likelihood/" )
include_directories( "src/models/" )
include_directories( "src/features/" )
include_directories( "src/libs/libhungarian/" )

if(EXISTS "/usr/local/cuda/include")
include_directories( "/usr/local/cuda/include")

add_executable( mtt_dpp src/test_dpp.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/dpp.cpp src/detectors/hog_detector.cpp src/detectors/cuda_hog_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_dpp ${OpenCV_LIBS} -lpthread)
target_compile_definitions( mtt_dpp PRIVATE -DWITH_CUDA )

add_executable( mtt_phd src/test_phd.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/features/haar.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_phd ${OpenCV_LIBS} -lpthread)
target_compile_definitions( mtt_phd PRIVATE -DWITH_CUDA )

add_executable( mtt_dpp_pets src/test_dpp_pets.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/detectors/hog_detector.cpp src/detectors/cuda_hog_detector.cpp src/dpp.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_dpp_pets ${OpenCV_LIBS} -lpthread)
target_compile_definitions( mtt_dpp_pets PRIVATE -DWITH_CUDA )

add_executable( mtt_phd_pets src/test_phd_pets.cpp src/detectors/hog_detector.cpp src/detectors/cuda_hog_detector.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_phd_pets ${OpenCV_LIBS} -lpthread)
target_compile_definitions( mtt_phd_pets PRIVATE -DWITH_CUDA )

else()

add_executable( mtt_dpp src/test_dpp.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/dpp.cpp src/detectors/hog_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_dpp ${OpenCV_LIBS} -lpthread)

add_executable( mtt_phd src/test_phd.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/detectors/hog_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_phd ${OpenCV_LIBS} -lpthread)

add_executable( mtt_dpp_pets src/test_dpp_pets.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/detectors/hog_detector.cpp src/dpp.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_dpp_pets ${OpenCV_LIBS} -lpthread)

add_executable( mtt_phd_pets src/test_phd_pets.cpp src/detectors/hog_detector.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( mtt_phd_pets ${OpenCV_LIBS} -lpthread)


endif()