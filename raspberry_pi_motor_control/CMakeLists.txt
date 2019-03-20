cmake_minimum_required(VERSION 2.6)

project(RPi_Motor_Control)
add_definitions(-std=c++11 -lwiringPi -lpthread)
include_directories(include)

file(GLOB SRC src/*)

message(${SRC})
add_executable(motor_control ${SRC})
target_link_libraries(motor_control wiringPi)
