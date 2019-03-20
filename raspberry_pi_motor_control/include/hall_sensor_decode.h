// Copyright 2019 Tsung-Han Lee
/**************************************
 ** FILE    : hall_sensor_decode.h
 ** AUTHOR  : Tsung-Han Lee
 **************************************/
#ifndef HALL_SENSOR_DECODE_H_
#define HALL_SENSOR_DECODE_H_

#include <wiringPi.h>

/* TODO... mapping BCM Pin to WiringPi Pin number */
#define LEFT_HALL_A     1
#define LEFT_HALL_B     2
#define RIGHT_HALL_A    3
#define RIGHT_HALL_B    4
/*
extern int LEFT_HALL_A;
extern int LEFT_HALL_B;
extern int RIGHT_HALL_A;
extern int RIGHT_HALL_B;
*/

/********************************************************
 *  WheelState should be defined as volatile since it
 *  will be modified in ISR
 ********************************************************/
typedef volatile struct WheelState {
    volatile unsigned int numStateChange;
    volatile bool direction;
    volatile unsigned short state;
    volatile unsigned short prestate;
    volatile int hallA;
    volatile int hallB;

    WheelState () {
        numStateChange  = 0;
        direction       = false;
        state           = 0;
        prestate        = 0;
        hallA           = 0;
        hallB           = 0;
    }
} WheelState_t;


void hallSensor_ISR(void);
void closeHallSensor();
WheelState_t* const leftWheel = new WheelState_t();
WheelState_t* const rightWheel = new WheelState_t();

#endif  //
