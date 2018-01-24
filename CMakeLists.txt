CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( mtt )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11 -fopenmp")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g -traceback")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( "/usr/include/eigen3/" "/usr/local/include/eigen3/" )
INCLUDE_DIRECTORIES( "src/utils/" )
INCLUDE_DIRECTORIES( "src/likelihood/" )
INCLUDE_DIRECTORIES( "src/models/" )
INCLUDE_DIRECTORIES( "src/features/" )
INCLUDE_DIRECTORIES( "src/libs/libhungarian/" )
INCLUDE_DIRECTORIES( "src/libs/nms/" )

add_subdirectory( "src/libs/piotr_fhog/" )

OPTION(MAKE_GM_PHD "Make GM PHD" ON)

SET( COMMON_SOURCES
    src/utils/image_generator.cpp
    src/utils/sequential_image_generator.cpp
    src/utils/utils.cpp
    src/utils/c_utils.cpp
    src/em.cpp
    src/libs/libhungarian/hungarian.c
)

if(MAKE_LR_DETECTOR)
    add_executable(lr_detector src/test_lr_detector.cpp src/detectors/CPU_LR_hog_detector.cpp src/likelihood/CPU_logistic_regression.cpp src/likelihood/logistic_regression.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/dpp/dpp.cpp ${COMMON_SOURCES})
    target_link_libraries( lr_detector ${OpenCV_LIBS} fhog)
endif()

if(MAKE_GM_PHD)
    add_executable( gm_phd src/test_gm_phd.cpp src/models/phd_gaussian_mixture.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( gm_phd ${OpenCV_LIBS} -lpthread)
    #target_compile_definitions( gm_phd PRIVATE -DWITH_SEQUENTIAL_IMAGE_GENERATOR )
    target_compile_definitions( gm_phd PRIVATE )
endif()


