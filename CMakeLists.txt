cmake_minimum_required(VERSION 3.16)
project(clustering_kmeans_project C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(kmeans STATIC
    src/kmeans.c
    src/csv.c
)

target_link_libraries(kmeans PUBLIC m)

target_include_directories(kmeans PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(kmeans_demo
    demo/kmeans_demo.c
)

target_include_directories(kmeans_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(kmeans_demo PRIVATE kmeans)

