cmake_minimum_required (VERSION 3.3)

project (llama-reflection)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} reflection.cpp)
target_link_libraries(${PROJECT_NAME} llama::llama)
target_compile_options(llama-reflection PRIVATE -std=c++2a -freflection)

install(FILES "${PROJECT_BINARY_DIR}/llama-reflection" DESTINATION "bin")
