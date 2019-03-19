// Copyright 2019 Tsung-Han Lee
/**************************************
 ** FILE    : motorControl.cpp
 ** AUTHOR  : Tsung-Han Lee
 **************************************/

#include <errno.h>
#include <string.h>
#include <iostream>

#include "hall_sensor_decode.h"

/********************* PARAMETERS ***************************************/
const int leftPWM   = 1;      // GPIO 1 for wiringPi = GPIO 18 for BCM
const int rightPWM  = 23;    // GPIO 23 for wiringPi = GPIO 13 for BCM
const int rightPWMn = 24;   // GPIO 24 for wiringPi  = GPIO 19 for BCM
const int leftPWMn  = 26;  // GPIO 26 for wiringPi   = GPIO 12 for BCM

double leftDutyRatio  = 0.0;
double rightDutyRatio = 0.0;
/***********************************************************************/

int initHallSensors();
int initPWM();

int main(int argc, char* argv[]) {
    /* Setup wiringPi */
    if (wiringPiSetup() < 0) {
        std::cerr << "Unable to setup wiringPi: "
                  << strerror(errno) << std::endl;
        return -1;
    }

    if (initHallSensors() < 0) {
        std::cerr << "Init hall sensors failed!" << std::endl;
    }

    return 0;
}

int initHallSensors() {
    /* Register ISR while hall sensor rising and falling */
    if (wiringPiISR(LEFT_HALL_A, INT_EDGE_BOTH, &hallSensor_ISR) < 0) {
        std::cerr << "Unable to setup ISR: "
                  << strerror(errno) << std::endl;
        return -1;
    }
    /* Register ISR while hall sensor rising and falling */
    if (wiringPiISR(LEFT_HALL_B, INT_EDGE_BOTH, &hallSensor_ISR) < 0) {
        std::cerr << "Unable to setup ISR: "
                  << strerror(errno) << std::endl;
        return -1;
    }
    /* Register ISR while hall sensor rising and falling */
    if (wiringPiISR(RIGHT_HALL_A, INT_EDGE_BOTH, &hallSensor_ISR) < 0) {
        std::cerr << "Unable to setup ISR: "
                  << strerror(errno) << std::endl;
        return -1;
    }
    /* Register ISR while hall sensor rising and falling */
    if (wiringPiISR(RIGHT_HALL_B, INT_EDGE_BOTH, &hallSensor_ISR) < 0) {
        std::cerr << "Unable to setup ISR: "
                  << strerror(errno) << std::endl;
        return -1;
    }

    return 0;
}

int initPWM() {
    /* set pwm pin as output */
    pinMode(leftPWM, PWM_OUTPUT);
    pinMode(leftPWMn, PWM_OUTPUT);
    pinMode(rightPWM, PWM_OUTPUT);
    pinMode(rightPWMn, PWM_OUTPUT);

    pwmWrite(leftPWM, static_cast<int>(1024*0.5));
    pwmWrite(rightPWM, static_cast<int>(1024*0.5));
}
