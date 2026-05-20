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

option(ANTONINA_GUI "Build with SFML GUI viewer" ON)


if(ANTONINA_GUI)
    find_package(SFML 3 COMPONENTS Graphics Window System QUIET)

    if(NOT SFML_FOUND)
        message(STATUS "SFML 3 not found in system, fetching from GitHub...")
        include(FetchContent)
        set(SFML_BUILD_AUDIO   OFF CACHE BOOL "" FORCE)
        set(SFML_BUILD_NETWORK OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS  ON  CACHE BOOL "" FORCE)
        FetchContent_Declare(
            SFML
            GIT_REPOSITORY https://github.com/SFML/SFML.git
            GIT_TAG        3.0.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(SFML)
    else()
        message(STATUS "Using system SFML ${SFML_VERSION}")
    endif()
endif()


set(CORE_SOURCES
    Layer.cpp
    Perceptron.cpp
    NeuroEvolution.cpp
    NeatEvolution.cpp
    TestGenerator.cpp
    AntoninaAPI.cpp
)

set(GUI_SOURCES
    Viewer.cpp
)

add_executable(antonina_ai
    ${CORE_SOURCES}
    main.cpp
)

if(ANTONINA_GUI)
    target_sources(antonina_ai PRIVATE ${GUI_SOURCES})
    target_link_libraries(antonina_ai PRIVATE SFML::Graphics SFML::Window SFML::System)
    set_target_properties(antonina_ai PROPERTIES WIN32_EXECUTABLE TRUE)
    if(MSVC)
        target_link_options(antonina_ai PRIVATE /ENTRY:mainCRTStartup)
    endif()
else()
    target_compile_definitions(antonina_ai PRIVATE NO_GUI=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(antonina_ai PRIVATE Threads::Threads)

if(WIN32 AND ANTONINA_GUI AND TARGET SFML::Graphics)
    add_custom_command(TARGET antonina_ai POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:antonina_ai> $<TARGET_FILE_DIR:antonina_ai>
        COMMAND_EXPAND_LISTS
    )
endif()


add_executable(test_equiv
    ${CORE_SOURCES}
    test_equiv.cpp
)
target_link_libraries(test_equiv PRIVATE Threads::Threads)
target_compile_definitions(test_equiv PRIVATE NO_GUI=1)

add_executable(diagnose_neat
    ${CORE_SOURCES}
    diagnose_neat.cpp
)
target_link_libraries(diagnose_neat PRIVATE Threads::Threads)
target_compile_definitions(diagnose_neat PRIVATE NO_GUI=1)

add_executable(generate_tests
    TestGenerator.cpp
    generate_tests.cpp
)
