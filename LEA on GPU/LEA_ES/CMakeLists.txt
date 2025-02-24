cmake_minimum_required(VERSION 3.18)
project(CUDAProject LANGUAGES CXX CUDA)

# CUDA 표준 설정
set(CMAKE_CUDA_STANDARD 14)

# CUDA optimization level
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# CMakeLists.txt
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA debug
# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")


# 소스 파일 추가
set(SOURCES
    kernel.cu
    lea_set.cu
    lea_es.cu
)

# 헤더 파일 경로 추가 (필요 시)
include_directories(
    .
)

# 실행 파일 생성
add_executable(main ${SOURCES})

# GPU 아키텍처 설정 (예: RTX 3060은 Compute Capability 8.6)
set_target_properties(
    main
    PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION ON
)
