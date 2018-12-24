CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( mtt )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11")
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
INCLUDE_DIRECTORIES( "src//" )
INCLUDE_DIRECTORIES( "src/libs/libhungarian/" )
INCLUDE_DIRECTORIES( "src/libs/nms/" )
INCLUDE_DIRECTORIES( "src/dpp/")

OPTION(MAKE_GM_PHD "Make GM PHD" OFF)
OPTION(MAKE_DPP "Make DPP" OFF)
OPTION(MAKE_YOLO "Make YOLO" ON)
OPTION(MAKE_YOLO_WEBCAM "Make YOLO" OFF)
OPTION(MAKE_PHD "Make PHD" OFF)

SET( UTIL_SOURCES
    src/utils/utils.cpp
    src/utils/image_generator.cpp
    src/utils/c_utils.cpp
)

SET( COMMON_SOURCES
    ${UTIL_SOURCES}
    src/em.cpp
    src/libs/libhungarian/hungarian.c
)

if(MAKE_GM_PHD)
    add_executable( gm_phd src/test_gm_phd.cpp src/models/phd_gaussian_mixture.cpp src/dpp/dpp.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( gm_phd ${OpenCV_LIBS} -lpthread)
endif()

if(MAKE_DPP)
    add_executable( dpp src/test_dpp.cpp src/dpp/dpp.cpp src/models/phd_gaussian_mixture.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( dpp ${OpenCV_LIBS} -lpthread)
endif()

if(MAKE_PHD)
    add_executable( phd src/test_phd.cpp src/models/phd_particle_filter.cpp src/detectors/yolo_detector.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( phd ${OpenCV_LIBS} -lpthread)
endif()

if(MAKE_YOLO)
    add_executable( yolo src/test_yolo.cpp  src/dpp/dpp.cpp src/detectors/yolo_detector.cpp src/detectors/hog_detector.cpp src/models/phd_gaussian_mixture.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( yolo ${OpenCV_LIBS} -lpthread)

    add_executable( yolo_dpp src/test_yolo_dpp.cpp  src/dpp/dpp.cpp src/detectors/yolo_detector.cpp src/detectors/hog_detector.cpp src/models/phd_gaussian_mixture.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( yolo_dpp ${OpenCV_LIBS} -lpthread)

    add_executable( yolo_nms src/test_yolo_nms.cpp  src/dpp/dpp.cpp src/detectors/yolo_detector.cpp src/detectors/hog_detector.cpp src/models/phd_gaussian_mixture.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( yolo_nms ${OpenCV_LIBS} -lpthread)
endif()

if(MAKE_YOLO_WEBCAM)
    add_executable( yolo_webcam src/test_yolo_webcam.cpp  src/dpp/dpp.cpp src/detectors/yolo_detector.cpp src/detectors/hog_detector.cpp src/models/phd_gaussian_mixture.cpp src/likelihood/gaussian.cpp src/likelihood/multivariate_gaussian.cpp ${COMMON_SOURCES} )
    target_link_libraries( yolo_webcam ${OpenCV_LIBS} -pthread)
endif()