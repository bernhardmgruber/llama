cmake_minimum_required (VERSION 3.15)
project(llama-nbody)

find_package(Vc QUIET)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()

add_executable(${PROJECT_NAME} nbody.cpp ../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
if (Vc_FOUND)
	target_link_libraries(${PROJECT_NAME} PRIVATE Vc::Vc)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-fno-math-errno # sqrt prevents vectorization otherwise
		-march=native
		-ffast-math
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
		/fp:fast
	)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	# GCC will try to find TBB and if it is installed use it for the parallel STL
	# unfortunately, it does not link against TBB automatically
	find_package(TBB CONFIG QUIET)
	if (TBB_FOUND)
		# TBB needs pthreads
		find_package(Threads REQUIRED)
		target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb Threads::Threads)
	endif()
endif()
