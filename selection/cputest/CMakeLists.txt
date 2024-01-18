cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(selection_cpu_cmake)
#set(CMAKE_PREFIX_PATH "/mnt/ssd1/rujie/pytorch/C++/libtorch")

find_package(Python3 COMPONENTS Interpreter Development)
include_directories(/home/rujie/anaconda3/envs/torch/include/python3.8)
#find_package(OpenCV REQUIRED)
find_package(Torch QUIET REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_PREFIX_PATH "$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')")
# set(CMAKE_PREFIX_PATH "/home/rujie/anaconda3/envs/torch/lib/python3.8/site-packages/torch/share/cmake")
add_library(selection_cpu SHARED selection_cpu.cpp)
#add_executable(selection_cpu_cmake selection_cpu.cpp)
#target_compile_features(selection_cpu PRIVATE cxx_std_14)
target_link_libraries(selection_cpu "${TORCH_LIBRARIES}")
set_property(TARGET selection_cpu PROPERTY CXX_STANDARD 14)
#target_link_libraries(selection_cpu opencv_core opencv_imgproc)




