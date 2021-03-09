cmake_minimum_required (VERSION 3.15)
project(llama-hep_analysis)

set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} hep_analysis.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
