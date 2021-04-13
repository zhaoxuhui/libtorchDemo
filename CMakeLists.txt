cmake_minimum_required(VERSION 3.15)
project(libtorchDemo)

set(CMAKE_CXX_STANDARD 11)

# 由于两个Demo用到了不同版本的libtorch，所以只能依次编译
# 一定要记得，切换后清空一下CMake的缓存文件，不然会报错
# 另外，记得将libtorch替换成你自己的路径

## -------------------------------------------------------------------------------------------------
## libtorch 1.3.1 - 针对libtorchDemo
#set(Torch_DIR /root/Softwares/libtorch/share/cmake/Torch)
#find_package(Torch REQUIRED)
#include_directories(${TORCH_INCLUDE_DIRS})
#
#add_executable(libtorchDemo main.cpp)
#target_link_libraries(libtorchDemo ${TORCH_LIBRARIES})
## -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# libtorch 1.0.0 - 针对lowlightEnhanceDemo
set(TORCH_PATH /root/Softwares/pytorch/torch/lib/tmp_install/share/cmake/Torch)
if( TORCH_PATH )
    message("TORCH_PATH set to: ${TORCH_PATH}")
    set(Torch_DIR ${TORCH_PATH})
else()
    message(FATAL_ERROR "Need to specify Torch path, e.g., pytorch/torch/share/cmake/Torch ")
endif()

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message(STATUS "Torch version is: ${Torch_VERSION}")

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_DIRS})

add_executable(lowlightEnhanceDemo lowlight.cpp)
target_link_libraries(lowlightEnhanceDemo ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# -------------------------------------------------------------------------------------------------





