cmake_minimum_required(VERSION 3.22)
project(antonina_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /EHsc /W3)
else()
    add_compile_options(-O3 -march=native -Wall)
endif()

find_package(SFML 3 COMPONENTS Graphics Window System QUIET)
if(NOT SFML_FOUND)
    message(STATUS "SFML 3 not found in system, fetching from GitHub...")
    include(FetchContent)
    set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
    set(SFML_BUILD_NETWORK OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    FetchContent_Declare(SFML GIT_REPOSITORY https://github.com/SFML/SFML.git GIT_TAG 3.0.0 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SFML)
else()
    message(STATUS "Using system SFML ${SFML_VERSION}")
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(PROJECT_INCLUDE_DIRS
    ${SRC_DIR}/app
    ${SRC_DIR}/core
    ${SRC_DIR}/game
    ${SRC_DIR}/gui
    ${SRC_DIR}/training
)

set(CORE_SOURCES
    ${SRC_DIR}/core/Layer.cpp
    ${SRC_DIR}/core/NeatGenome.cpp
    ${SRC_DIR}/core/NeatInnovation.cpp
    ${SRC_DIR}/core/Perceptron.cpp
    ${SRC_DIR}/game/AntoninaAPI.cpp
    ${SRC_DIR}/game/TestGenerator.cpp
    ${SRC_DIR}/training/NeatEvolution.cpp
    ${SRC_DIR}/training/NeuroEvolution.cpp
    ${SRC_DIR}/training/TestWeightTable.cpp
    ${SRC_DIR}/training/ThreadPool.cpp
    ${SRC_DIR}/training/TrainingSettings.cpp
    ${SRC_DIR}/training/TrainingSettingsJson.cpp
    ${SRC_DIR}/training/TrainingUtils.cpp
)

set(GUI_SOURCES
    ${SRC_DIR}/gui/ScopedViewerLogRedirect.cpp
    ${SRC_DIR}/gui/Viewer.cpp
    ${SRC_DIR}/gui/ViewerLogStreamBuf.cpp
)

add_executable(antonina_ai
    ${CORE_SOURCES}
    ${GUI_SOURCES}
    ${SRC_DIR}/app/main.cpp
)
target_include_directories(antonina_ai PRIVATE ${PROJECT_INCLUDE_DIRS})
find_package(Threads REQUIRED)
target_link_libraries(antonina_ai PRIVATE Threads::Threads SFML::Graphics SFML::Window SFML::System)
set_target_properties(antonina_ai PROPERTIES WIN32_EXECUTABLE TRUE)
if(MSVC)
    target_link_options(antonina_ai PRIVATE /ENTRY:mainCRTStartup)
endif()

if(WIN32 AND TARGET SFML::Graphics)
    add_custom_command(TARGET antonina_ai POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:antonina_ai> $<TARGET_FILE_DIR:antonina_ai>
        COMMAND_EXPAND_LISTS
    )
endif()

add_executable(generate_tests
    ${SRC_DIR}/game/TestGenerator.cpp
    ${SRC_DIR}/app/generate_tests.cpp
)
target_include_directories(generate_tests PRIVATE ${PROJECT_INCLUDE_DIRS})
