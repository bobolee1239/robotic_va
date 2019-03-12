cmake_minimum_required(VERSION 2.6)

project(robotic_va_motion_control)

add_definitions(-std=c++98 -lwiringPi -lpthread)
file(GLOB FILES src/*.cpp)

add_executable(pwm ${FILES})
target_link_libraries(pwm wiringPi)

