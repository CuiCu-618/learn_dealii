CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

PROJECT(MatrixFree)

include_directories(/usr/local/bin/../include/)
link_directories(/usr/local/bin/../lib/)
add_definitions(-DLIKWID_PERFMON)


find_package(likwid REQUIRED COMPONENTS marker OPTIONAL_COMPONENTS nvmarker
              HINTS /usr/local/lib/)

message(STATUS "")
message(STATUS "likwid_FOUND: ${likwid_FOUND}")

foreach(_VAR LIKWID_INCLUDES LIKWID_LIBRARIES LIKWID_ROOT_DIR
    LIKWID_NVIDIA_INTERFACE LIKWID_FORTRAN_INTERFACE
    LIKWID_INCLUDE_DIR LIKWID_LIBRARY LIKWID_INCLUDE_DIRS
    LIKWID_LIBRARIES LIKWID_LIBRARY_DIRS)
    message(STATUS "${_VAR}: ${${_VAR}}")
endforeach()

foreach(_VAR ${LIKWID_EXECUTABLE_OPTIONS})
    message(STATUS "LIKWID_${_VAR}_EXECUTABLE: ${LIKWID_${_VAR}_EXECUTABLE}")
endforeach()

foreach(_VAR ${LIKWID_COMPONENT_OPTIONS})
    message(STATUS "likwid_${_VAR}_FOUND: ${likwid_${_VAR}_FOUND}")
endforeach()
message(STATUS "")

FIND_PACKAGE(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ 
  /export/home/cucui/software/dealii/build/
  /export/home/cucui/CLionProjects/dealii-9.3.1 $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_EXECUTABLE(main main.cc)
TARGET_LINK_LIBRARIES(main likwid)
DEAL_II_SETUP_TARGET(main)
