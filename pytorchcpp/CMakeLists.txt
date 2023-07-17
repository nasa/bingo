cmake_minimum_required(VERSION 3.19)
project(sandbox)

set(Torch_DIR ${PROJECT_SOURCE_DIR}/external/libtorch/share/cmake/Torch)
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/external/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(sandbox main.cpp)
include_directories(${PROJECT_SOURCE_DIR}/external/libtorch)

target_link_libraries(sandbox "${TORCH_LIBRARIES}")
set(CMAKE_CXX_STANDARD 14)
set_property(TARGET sandbox PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET sandbox
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:sandbox>)
endif (MSVC)
