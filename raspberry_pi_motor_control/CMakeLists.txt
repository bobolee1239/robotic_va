cmake_minimum_required(VERSION 2.6);

project(RPi_Motor_Control);

file(GLOB SRC src/*);

include_directories(include);

message(${SRC});
