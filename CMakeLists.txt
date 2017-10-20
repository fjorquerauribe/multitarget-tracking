cmake_minimum_required(VERSION 2.8)
project( mtt )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS               "-Wall -std=c++11 -fopenmp")
#SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g -traceback")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
find_package( OpenCV REQUIRED )
find_package( CUDA QUIET REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories( "/usr/include/eigen3/" )
include_directories( "/usr/include/eigen3/" "/usr/local/include/eigen3/" )
include_directories( "src/utils/" )
include_directories( "src/likelihood/" )
include_directories( "src/models/" )
include_directories( "src/features/" )
include_directories( "src/libs/libhungarian/" )

if(CUDA_FOUND)
include_directories( ${CUDA_TOOLKIT_INCLUDE} )

add_executable( phd src/test_phd.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/likelihood/logistic_regression.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( phd ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} -lpthread)
target_compile_definitions( phd PRIVATE -DWITH_CUDA )

#add_executable( phd_pets src/test_phd_pets.cpp src/detectors/cuda_hog_detector.cpp src/likelihood/logistic_regression.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
#target_link_libraries( phd_pets ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} -lpthread)
#target_compile_definitions( phd_pets PRIVATE -DWITH_CUDA )

#add_executable( dpp src/test_dpp.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/dpp.cpp src/detectors/cuda_hog_detector.cpp src/likelihood/logistic_regression.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
#target_link_libraries( dpp ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} -lpthread)
#target_compile_definitions( dpp PRIVATE -DWITH_CUDA )

#add_executable( dpp_pets src/test_dpp_pets.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/detectors/cuda_hog_detector.cpp src/likelihood/logistic_regression.cpp src/dpp.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
#target_link_libraries( dpp_pets ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} -lpthread)
#target_compile_definitions( dpp_pets PRIVATE -DWITH_CUDA )

else()

add_executable( phd src/test_phd.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/detectors/hog_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( phd ${OpenCV_LIBS} -lpthread)

add_executable( phd_pets src/test_phd_pets.cpp src/detectors/hog_detector.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( phd_pets ${OpenCV_LIBS} -lpthread)

add_executable( dpp src/test_dpp.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/dpp.cpp src/detectors/hog_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( dpp ${OpenCV_LIBS} -lpthread)

add_executable( dpp_pets src/test_dpp_pets.cpp src/utils/image_generator.cpp src/utils/utils.cpp src/detectors/hog_detector.cpp src/dpp.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp src/em.cpp src/libs/libhungarian/hungarian.c)
target_link_libraries( dpp_pets ${OpenCV_LIBS} -lpthread)

endif()