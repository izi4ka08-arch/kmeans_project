cmake_minimum_required(VERSION 3.16)
project(clustering_kmeans_project C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(kmeans STATIC
    src/kmeans.c
    src/csv.c
)

if(UNIX AND NOT APPLE)
    target_link_libraries(kmeans PUBLIC m)
endif()

target_include_directories(kmeans PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(clustering_demo
    demo/clustering_demo.c
)

target_include_directories(clustering_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(clustering_demo PRIVATE kmeans)

