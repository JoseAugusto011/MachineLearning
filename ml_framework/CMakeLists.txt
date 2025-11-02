cmake_minimum_required(VERSION 3.10)
project(MLFramework)

set(CMAKE_CXX_STANDARD 14)

# Encontra Eigen
find_package(Eigen3 REQUIRED)

# Configura include directories - CORRIGIDO
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Arquivos fonte
set(SOURCES
    src/PocketPLA.cpp
    src/Metrics.cpp
    main.cpp
)

# Executável principal
add_executable(ml_test ${SOURCES})

# Link com Eigen
target_link_libraries(ml_test Eigen3::Eigen)