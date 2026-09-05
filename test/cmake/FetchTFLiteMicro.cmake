# Fetches TFLite Micro (and the third-party headers it needs) at configure
# time instead of vendoring a copy of them into this repo. Pinned to the
# exact versions/commits TFLite Micro's own Makefile-based build downloads
# (tensorflow/lite/micro/tools/make/third_party_downloads.inc in the fetched
# source), including the one patch that's actually required (flatbuffers).
#
# TFLite Micro itself ships no CMakeLists.txt (its own build is
# Bazel/Make-based) -- FetchContent_Populate() just fetches the source, it
# doesn't try to configure/build it as a CMake subproject. Defines
# TFLM_SOURCE_DIR and the *_INCLUDE_DIR variables test/CMakeLists.txt uses.
include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(TFLITE_MICRO_GIT_TAG bfeb43aa795adad4044bbf2e709062b1b989a98a)
FetchContent_Declare(tflite_micro
  URL https://github.com/tensorflow/tflite-micro/archive/${TFLITE_MICRO_GIT_TAG}.zip
)
FetchContent_GetProperties(tflite_micro)
if(NOT tflite_micro_POPULATED)
  FetchContent_Populate(tflite_micro)
endif()
set(TFLM_SOURCE_DIR ${tflite_micro_SOURCE_DIR})

FetchContent_Declare(gemmlowp_dl
  URL https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip
)
FetchContent_GetProperties(gemmlowp_dl)
if(NOT gemmlowp_dl_POPULATED)
  FetchContent_Populate(gemmlowp_dl)
endif()
set(GEMMLOWP_INCLUDE_DIR ${gemmlowp_dl_SOURCE_DIR})

FetchContent_Declare(ruy_dl
  URL https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip
)
FetchContent_GetProperties(ruy_dl)
if(NOT ruy_dl_POPULATED)
  FetchContent_Populate(ruy_dl)
endif()
set(RUY_INCLUDE_DIR ${ruy_dl_SOURCE_DIR})

# flatbuffers needs TFLM's patch (base.h/default_allocator.h/
# flatbuffer_builder.h/flexbuffers.h/util.h) -- apply it once, right after
# fetching, rather than on every configure (FetchContent_Populate would
# otherwise re-run the PATCH_COMMAND and fail on an already-patched tree).
FetchContent_Declare(flatbuffers_dl
  URL https://github.com/google/flatbuffers/archive/refs/tags/v25.9.23.zip
)
FetchContent_GetProperties(flatbuffers_dl)
if(NOT flatbuffers_dl_POPULATED)
  FetchContent_Populate(flatbuffers_dl)
endif()
# -N/--forward: skip hunks that look already applied instead of prompting --
# needed because a re-configure (e.g. after editing this file) can re-enter
# this branch against a source dir FetchContent already populated (and this
# script already patched) on a previous run within the same build dir.
execute_process(
  COMMAND patch -p1 -N -i ${TFLM_SOURCE_DIR}/tensorflow/lite/micro/tools/make/flatbuffers.patch
  WORKING_DIRECTORY ${flatbuffers_dl_SOURCE_DIR}
  RESULT_VARIABLE _flatbuffers_patch_result
  OUTPUT_QUIET
)
if(_flatbuffers_patch_result GREATER 1)
  message(FATAL_ERROR "Failed to apply TFLite Micro's flatbuffers patch (exit ${_flatbuffers_patch_result})")
endif()
set(FLATBUFFERS_INCLUDE_DIR ${flatbuffers_dl_SOURCE_DIR}/include)
