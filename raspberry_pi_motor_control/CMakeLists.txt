cmake_minimum_required(VERSION 2.6)

project(RPi_Motor_Control)
add_definitions(-std=c++11 -lwiringPi -lpthread)
include_directories(include)

set(MOTOR_CONTROL src/motorControl.cpp)
set(TEST_RPM src/test.cpp)

add_executable(motor_control ${MOTOR_CONTROL})
target_link_libraries(motor_control wiringPi)

add_executable(test_rpm ${TEST_RPM})
target_link_libraries(test_rpm wiringPi)

add_executable(simple_vehicle src/simple_vehicle.cpp)
target_link_libraries(simple_vehicle wiringPi)
