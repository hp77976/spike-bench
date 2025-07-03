cmake_minimum_required(VERSION 4.0)
project(machine CXX)
#project(machine CXX CUDA)

#enable_language(CUDA)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	message("Found ccache!")
endif()

set(CMAKE_CXX_COMPILER /bin/clang++ CACHE STRING "")
set(CMAKE_C_COMPILER /bin/clang CACHE STRING "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_CUDA_STANDARD 20)
#set(CMAKE_CUDA_COMPILER nvcc)

set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(/usr/include/cpp)
include_directories(/usr/include/)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

option(BUILD_UI "enables the UI" ON)

include(GNUInstallDirs)

if(BUILD_UI)
	set(RLIMGUI_PATH ext/rlImGui/)
	file(GLOB RLIMGUI_GLOB
		${RLIMGUI_PATH}/imgui_impl_raylib.h
		${RLIMGUI_PATH}/rlImGuiColors.h
		${RLIMGUI_PATH}/rlImGui.h
		${RLIMGUI_PATH}/rlImGui.cpp
	)

	add_library("rlimgui" STATIC ${RLIMGUI_GLOB})
	target_include_directories("rlimgui" PUBLIC ${RLIMGUI_PATH})
	target_link_libraries("rlimgui" PRIVATE glfw)

	set(IMGUI_PATH ext/imgui)
	file(GLOB IMGUI_GLOB
		${IMGUI_PATH}/imgui.h
		${IMGUI_PATH}/imgui.cpp
		${IMGUI_PATH}/imconfig.h
		${IMGUI_PATH}/imgui_demo.cpp
		${IMGUI_PATH}/imgui_draw.cpp
		${IMGUI_PATH}/imgui_internal.h
		${IMGUI_PATH}/imstb_rectpack.h
		${IMGUI_PATH}/imstb_textedit.h
		${IMGUI_PATH}/imstb_truetype.h
		${IMGUI_PATH}/imgui_tables.cpp
		${IMGUI_PATH}/imgui_widgets.cpp

		# specific bindings...
		${IMGUI_PATH}/backends/imgui_impl_sdl2.h
		${IMGUI_PATH}/backends/imgui_impl_sdl2.cpp
		${IMGUI_PATH}/backends/imgui_impl_opengl3.h
		${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp
		${IMGUI_PATH}/backends/imgui_impl_opengl3_loader.cpp)
	add_library("imgui" STATIC ${IMGUI_GLOB})
	target_include_directories("imgui" PUBLIC ${IMGUI_PATH})
	target_link_libraries("imgui" PRIVATE glfw)

	find_package(SDL2 REQUIRED)
	find_package(raylib REQUIRED)

	include_directories(${SDL2_INCLUDE_DIR})
	include_directories(${IMGUI_PATH})
endif()

#find_package(CUDA REQUIRED)
#find_package(CUDAToolkit REQUIRED)

#include_directories(${OpenMP_INCLUDE_DIRS})

#set(BLA_VENDER OpenBLAS)
#find_package(BLAS REQUIRED)
#link_directories(${BLAS_INCLUDE_DIRS})

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#	set(CMAKE_CUDA_ARCHITECTURES native)
#endif()

set(OPS
	""
)

message(${CMAKE_CXX_COMPILER_ID})

set(COMMON_FLAGS
	-std=c++20 -Wall -ffast-math
	-Wno-unused-variable
	-Wno-unused-but-set-variable
	-Wno-unknown-warning-option
	-Wno-nan-infinity-disabled
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
	set(COMMON_FLAGS ${COMMON_FLAGS} -fimf-use-svml=true -fp-model=fast -fvec-peel-loops)
endif()

set(STD_FLAGS ${COMMON_FLAGS} -O3)
set(STD_FLAGS ${STD_FLAGS} -g -mavx -mavx2 -mavx512f -mavx512vl -march=native)

set(DBG_FLAGS ${COMMON_FLAGS} -O0)
set(DBG_FLAGS ${DBG_FLAGS} -ggdb3 -mavx -mavx2 -mavx512f -mavx512vl -march=native)

set(ARR_FLAGS ${COMMON_FLAGS} -O1)
set(ARR_FLAGS ${ARR_FLAGS} -ggdb3 -mavx -mavx2 -mavx512f -mavx512vl -march=native)

set(VLG_FLAGS ${COMMON_FLAGS} -O0)
set(VLG_FLAGS ${VLG_FLAGS} -ggdb3 -march=broadwell)

remove_definitions(-DENABLE_SIMD)
remove_definitions(-DDEBUG_ARRAYS)
remove_definitions(-DDEBUG_TENSORS)
remove_definitions(-DENABLE_DEBUG_LOG)

set(TEST_SOURCES
	src/snn5/builder.cpp
	src/snn5/handler.cpp
	src/snn5/layout.cpp
	src/snn5/wrappers.cpp
	src/snn5/neurons.cpp
	src/snn5/synapses.cpp
	
	src/snn5/base.cpp
	src/snn5/snn.cpp
	src/snn5/conf.cpp
	src/snn5/log.cpp
	src/snn5/ui.cpp

	src/util/mnist.cpp
	src/util/debug.cpp
)

if(BUILD_UI)
	set(TEST_SOURCES ${TEST_SOURCES}
		src/rlimgui/rlImGui.cpp
		src/implot/implot.cpp
		src/implot/implot_items.cpp
	)
endif()

set(ALL_LIBS
	${BLAS_LIBRARIES}
)

if(BUILD_UI)
	set(ALL_LIBS ${ALL_LIBS}
		SDL2
		raylib
		imgui
	)
endif()

add_executable(test_bench src/test/bench.cpp ${TEST_SOURCES})
target_compile_options(test_bench PRIVATE ${STD_FLAGS})
target_compile_definitions(test_bench PUBLIC ENABLE_SIMD)
target_link_libraries(test_bench PRIVATE ${ALL_LIBS})

if(BUILD_UI)
	add_executable(test_vd2 src/test/vd2.cpp ${TEST_SOURCES})
	target_compile_options(test_vd2 PRIVATE ${VLG_FLAGS})
	target_compile_definitions(test_vd2 PUBLIC DEBUG_ARRAYS DEBUG_TENSORS)
	target_link_libraries(test_vd2 PRIVATE ${ALL_LIBS})

	add_executable(test_graph src/test/graph_stride.cpp ${TEST_SOURCES})
	target_compile_options(test_graph PRIVATE ${DBG_FLAGS})
	target_compile_definitions(test_graph PUBLIC ENABLE_SIMD)
	target_link_libraries(test_graph PRIVATE ${ALL_LIBS})

	add_executable(test_tensor src/test/tensor.cpp ${TEST_SOURCES})
	target_compile_options(test_tensor PRIVATE ${DBG_FLAGS})
	target_compile_definitions(test_tensor PUBLIC ENABLE_SIMD)
	target_link_libraries(test_tensor PRIVATE ${ALL_LIBS})

	add_executable(test_eigen src/test/eigen.cpp ${TEST_SOURCES})
	target_compile_options(test_eigen PRIVATE ${STD_FLAGS})
	target_compile_definitions(test_eigen PUBLIC ENABLE_SIMD)
	target_link_libraries(test_eigen PRIVATE ${ALL_LIBS})

	add_executable(test_ui2 src/test/test_ui2.cpp ${TEST_SOURCES})
	target_compile_options(test_ui2 PRIVATE ${STD_FLAGS})
	target_compile_definitions(test_ui2 PUBLIC ENABLE_SIMD)
	target_link_libraries(test_ui2 PRIVATE ${ALL_LIBS})

	add_executable(test_ui2_dbg src/test/test_ui2.cpp ${TEST_SOURCES})
	target_compile_options(test_ui2_dbg PRIVATE ${DBG_FLAGS})
	target_compile_definitions(
		test_ui2_dbg PUBLIC ENABLE_SIMD DEBUG_ARRAYS DEBUG_TENSORS ENABLE_DEBUG_LOG)
	target_link_libraries(test_ui2_dbg PRIVATE ${ALL_LIBS})

	add_executable(test_snn5_ui src/test/snn5_ui.cpp ${TEST_SOURCES})
	target_compile_options(test_snn5_ui PRIVATE ${STD_FLAGS})
	target_compile_definitions(test_snn5_ui PUBLIC ENABLE_SIMD)
	target_link_libraries(test_snn5_ui PRIVATE ${ALL_LIBS})

	add_executable(test_snn5_ui_arr src/test/snn5_ui.cpp ${TEST_SOURCES})
	target_compile_options(test_snn5_ui_arr PRIVATE ${ARR_FLAGS})
	target_compile_definitions(test_snn5_ui_arr PUBLIC ENABLE_SIMD DEBUG_ARRAYS DEBUG_TENSORS)
	target_link_libraries(test_snn5_ui_arr PRIVATE ${ALL_LIBS})

	add_executable(test_snn5_ui_dbg src/test/snn5_ui.cpp ${TEST_SOURCES})
	target_compile_options(test_snn5_ui_dbg PRIVATE ${DBG_FLAGS})
	target_compile_definitions(
		test_snn5_ui_dbg PUBLIC ENABLE_SIMD DEBUG_ARRAYS DEBUG_TENSORS ENABLE_DEBUG_LOG)
	target_link_libraries(test_snn5_ui_dbg PRIVATE ${ALL_LIBS})
endif()