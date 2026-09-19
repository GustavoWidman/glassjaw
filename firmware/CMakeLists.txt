# glassjaw firmware — esp-idf project for the esp32 + inmp441 detector.
# build: idf.py build   flash: idf.py flash monitor
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(glassjaw LANGUAGES C CXX)
