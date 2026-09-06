# Desktop CLI: ./tinytts "some text" -- MiniAudioStream output
# (arduino-audio-tools, fetched by the top-level CMakeLists.txt), text
# passed as argv[1]. Not part of the Arduino/ESP-IDF library surface at all
# (see DesktopMain.h's own doc) -- only built when -DTINYTTS_BUILD_DESKTOP_MAIN=ON.
#
#   cmake -S . -B build -DTINYTTS_BUILD_DESKTOP_MAIN=ON
#   cmake --build build
#   ./build/desktop/tinytts "Hello world!"
#
# The CMake *target* below stays named tinytts_desktop, not tinytts -- the
# top-level CMakeLists.txt already has a CMake target literally named
# `tinytts` (the header-only INTERFACE library every consumer links
# against), and target names must be unique within a project. OUTPUT_NAME
# below only renames the produced binary file, not the target itself.

# audio-tools doesn't vendor miniaudio.h itself (single-header
# github.com/mackron/miniaudio -- only the header matters here, we never
# use any of its own CMake-built targets: MiniAudioStream.h compiles the
# whole implementation directly via #define MINIAUDIO_IMPLEMENTATION) --
# every arduino-audio-tools example that uses MiniAudioStream keeps a
# local copy for the same reason.
#
# Deliberately FetchContent_Populate(), not FetchContent_MakeAvailable():
# the latter also runs add_subdirectory() on miniaudio's own CMakeLists.txt,
# which defines an installable libminiaudio.a target we never build or
# link -- `cmake --install` then fails outright looking for that file
# ("file INSTALL cannot find .../libminiaudio.a"). Populate-only avoids
# pulling in that project at all; this is the same pattern
# arduino-audio-tools' own top-level CMakeLists.txt already uses for its
# optional portaudio/arduino_emulator fetches, for the identical reason.
FetchContent_Declare(miniaudio
  GIT_REPOSITORY "https://github.com/mackron/miniaudio.git"
  GIT_TAG master)
FetchContent_GetProperties(miniaudio)
if(NOT miniaudio_POPULATED)
  FetchContent_Populate(miniaudio)
endif()

add_executable(tinytts_desktop main.cpp)
set_target_properties(tinytts_desktop PROPERTIES OUTPUT_NAME tinytts)
target_link_libraries(tinytts_desktop PRIVATE tinytts arduino-audio-tools)
target_include_directories(tinytts_desktop PRIVATE ${miniaudio_SOURCE_DIR})
# NO_MAIN: AudioTools/AudioLibs/Emulation/Main.h (pulled in transitively by
# IS_MIN_DESKTOP) otherwise defines its own int main(void) -- conflicts with
# this target's own main.cpp.
target_compile_definitions(tinytts_desktop PRIVATE IS_MIN_DESKTOP TINYTTS_DESKTOP_TASK NO_MAIN)

# Needed on Linux: std::thread (audio-tools' Desktop Task) links against
# pthread, and miniaudio's implementation needs libm/libdl for its backend
# probing.
find_package(Threads REQUIRED)
target_link_libraries(tinytts_desktop PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)

# Without this, `cmake --install`/`make install` silently installs nothing
# of ours at all (only arduino-audio-tools' own headers, as an incidental
# side effect of its unrelated install() rules) -- there's no default
# "install every executable" behavior in CMake.
install(TARGETS tinytts_desktop RUNTIME DESTINATION bin)
