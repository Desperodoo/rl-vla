cmake_minimum_required(VERSION 3.14)
project(inference_rtc VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 优化选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# 依赖
# ============================================================================

# ARX5 SDK
set(ARX5_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../arx5-sdk")
if(NOT EXISTS ${ARX5_SDK_DIR})
    message(FATAL_ERROR "ARX5 SDK not found at ${ARX5_SDK_DIR}")
endif()

message(STATUS "ARX5 SDK: ${ARX5_SDK_DIR}")

# 检测架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARX5_LIB_DIR "${ARX5_SDK_DIR}/lib/x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARX5_LIB_DIR "${ARX5_SDK_DIR}/lib/aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message(STATUS "ARX5 lib dir: ${ARX5_LIB_DIR}")

# 查找 pthread
find_package(Threads REQUIRED)

# Conda 环境中的依赖 (用于 arx5-sdk)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(spdlog REQUIRED)

# ============================================================================
# 头文件
# ============================================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared
    ${ARX5_SDK_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
)

# Conda 环境头文件 (KDL 等)
if(DEFINED ENV{CONDA_PREFIX})
    include_directories(
        $ENV{CONDA_PREFIX}/include
        $ENV{CONDA_PREFIX}/include/kdl_parser
        $ENV{CONDA_PREFIX}/include/urdfdom_headers
    )
endif()

# ============================================================================
# 库
# ============================================================================

# ARX5 库 (动态库)
add_library(arx5_hardware SHARED IMPORTED)
set_target_properties(arx5_hardware PROPERTIES
    IMPORTED_LOCATION "${ARX5_LIB_DIR}/libhardware.so"
)

add_library(arx5_solver SHARED IMPORTED)
set_target_properties(arx5_solver PROPERTIES
    IMPORTED_LOCATION "${ARX5_LIB_DIR}/libsolver.so"
)

# 设置 RPATH
set(CMAKE_INSTALL_RPATH "${ARX5_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# ============================================================================
# 可执行文件
# ============================================================================

add_executable(servo_main
    cpp/servo_main.cpp
)

target_link_libraries(servo_main
    arx5_hardware
    arx5_solver
    Eigen3::Eigen
    Threads::Threads
    spdlog::spdlog
    kdl_parser
    orocos-kdl
    rt  # 共享内存
)

# ============================================================================
# 安装
# ============================================================================

install(TARGETS servo_main
    RUNTIME DESTINATION bin
)

# ============================================================================
# 测试 (可选)
# ============================================================================

# 简单的插值器测试
add_executable(test_interpolator
    cpp/test_interpolator.cpp
)

target_link_libraries(test_interpolator
    Threads::Threads
)

# ============================================================================
# 打印配置
# ============================================================================

message(STATUS "")
message(STATUS "=== inference_rtc 配置 ===")
message(STATUS "  C++ 标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "  构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "  架构: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  ARX5 SDK: ${ARX5_SDK_DIR}")
message(STATUS "")
