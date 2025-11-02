cmake_minimum_required(VERSION 3.10)
project(MLFramework)

set(CMAKE_CXX_STANDARD 14)

# Encontra Eigen
find_package(Eigen3 REQUIRED)

# Configura include directories CORRETAMENTE
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Define diretório de source
set(SOURCE_DIR src)

# Arquivos fonte - ADICIONAR LinearRegression.cpp
# Adicionar LRClassifier.cpp à lista de sources
set(SOURCES
    ${SOURCE_DIR}/PocketPLA.cpp
    ${SOURCE_DIR}/Metrics.cpp
    ${SOURCE_DIR}/LinearRegression.cpp
    ${SOURCE_DIR}/LRClassifier.cpp  # NOVA LINHA
    main.cpp
)

# Executável principal
add_executable(ml_test ${SOURCES})

# Link com Eigen
target_link_libraries(ml_test Eigen3::Eigen)

# Configurações para melhor compilação
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")