CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( mtt )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11 -fopenmp")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g -traceback")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( CUDA QUIET REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( "/usr/include/eigen3/" "/usr/local/include/eigen3/" )
INCLUDE_DIRECTORIES( "src/utils/" )
INCLUDE_DIRECTORIES( "src/likelihood/" )
INCLUDE_DIRECTORIES( "src/models/" )
INCLUDE_DIRECTORIES( "src/features/" )
INCLUDE_DIRECTORIES( "src/libs/libhungarian/" )
INCLUDE_DIRECTORIES( "src/libs/nms/" )

add_subdirectory( "src/libs/piotr_fhog/" )

OPTION(MAKE_PHD "Make PHD" ON)
OPTION(MAKE_DPP "Make DPP" OFF)
OPTION(MAKE_LR_DETECTOR "Make LR_DETECTOR" OFF)

SET( COMMON_SOURCES
    src/utils/image_generator.cpp
    src/utils/utils.cpp
    src/utils/c_utils.cpp
    src/em.cpp
    src/libs/libhungarian/hungarian.c
)

if(MAKE_LR_DETECTOR)
    add_executable(lr_detector src/test_lr_detector.cpp src/detectors/CPU_LR_hog_detector.cpp src/likelihood/CPU_logistic_regression.cpp src/likelihood/logistic_regression.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES})
    target_link_libraries( lr_detector ${OpenCV_LIBS} fhog)
endif()

if(CUDA_FOUND)
    INCLUDE_DIRECTORIES( ${CUDA_TOOLKIT_INCLUDE} )

    if(MAKE_PHD)
        add_executable( phd src/test_phd.cpp src/likelihood/logistic_regression.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp ${COMMON_SOURCES})
        target_link_libraries( phd ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} -lpthread)
        target_compile_definitions( phd PRIVATE -DWITH_NMS )
    endif()

    if(MAKE_DPP)
        add_executable( dpp src/test_dpp.cpp src/dpp/dpp.cpp src/detectors/cuda_hog_detector.cpp src/likelihood/logistic_regression.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp ${COMMON_SOURCES})
        target_link_libraries( dpp ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} -lpthread)
        target_compile_definitions( dpp PRIVATE -DWITH_CUDA )
    endif()

else()

    if(MAKE_PHD)
        add_executable( phd src/test_phd.cpp src/detectors/hog_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp ${COMMON_SOURCES})
        target_link_libraries( phd ${OpenCV_LIBS} -lpthread)
    endif()

    if(MAKE_DPP)
        add_executable( dpp src/test_dpp.cpp src/dpp/dpp.cpp src/detectors/hog_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp src/models/phd_particle_filter.cpp ${COMMON_SOURCES})
        target_link_libraries( dpp ${OpenCV_LIBS} -lpthread)
    endif()

endif()