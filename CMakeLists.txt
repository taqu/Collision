cmake_minimum_required(VERSION 3.10)

function (IsGnuCompiler)
    if((${CMAKE_C_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
        set(COMPILER_IS_GNU ON PARENT_SCOPE)
    else()
        set(COMPILER_IS_GNU OFF PARENT_SCOPE)
    endif()
endfunction()

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(ProjectName bvh)
project(${ProjectName})

set(HEADERS "BinQBVH.h")
set(SOURCES "main.cpp;BinQBVH.cpp")

include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${SOURCES})

set(FILES ${HEADERS} ${SOURCES})

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

add_executable(${ProjectName} ${FILES})

IsGnuCompiler()

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_MSBC /W4 /WX- /nologo /fp:precise /arch:AVX2 /Zc:wchar_t /TP /Gd")

    if(MSVC_VERSION VERSION_LESS_EQUAL "1900")
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++latest")
    else()
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++17")
    endif()

    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1 /Gy /GR- /GS /Gm- /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Oi /GL /GR- /DNDEBUG /EHsc-")

    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")

elseif(COMPILER_IS_GNU)
    if((${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64") OR (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86"))
        message("x86_64 or x86")
        set(DEFAULT_CXX_FLAGS "-Wall -O2 -std=c++17 -std=gnu++17 -march=skylake")
    elseif((${CMAKE_SYSTEM_PROCESSOR} MATCHES "^armv7.*"))
        message("armv7")
        set(DEFAULT_CXX_FLAGS "-Wall -O3 -std=c++17 -std=gnu++17 -march=armv7-a -mlittle-endian")
    elseif((${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
        message("aarch64")
        set(DEFAULT_CXX_FLAGS "-Wall -O3 -std=c++17 -std=gnu++17 -march=armv8-a -mlittle-endian")
        #set(DEFAULT_CXX_FLAGS "-Wall -O2 -std=c++17 -std=gnu++17 -march=armv9-a")
    else()
        message("unsupported")
    endif()
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
elseif(XCODE)
endif()

set_target_properties(${ProjectName} PROPERTIES OUTPUT_NAME_DEBUG "${ProjectName}" OUTPUT_NAME_RELEASE "${ProjectName}")

