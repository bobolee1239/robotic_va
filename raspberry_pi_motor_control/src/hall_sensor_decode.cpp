// Copyright 2019 Tsung-Han Lee
/**************************************
 ** FILE    : hall_sensor_decode.cpp
 ** AUTHOR  : Tsung-Han Lee
 **************************************/

#include "hall_sensor_decode.h"

WheelState_t* const leftWheel, rightWheel;

void hallSensor_ISR(void) {
    /* read state of GPIO pin */
    leftWheel->hallA = digitalRead(LEFT_HALL_A);
    leftWheel->hallB = digitalRead(LEFT_HALL_B);

    /* hallA hallB should be 1, 0 */
    leftWheel->state = (leftWheel->hallA << 2)
                        + (leftWheel->hallA ^ leftWheel->hallB)
                        + 1;

    if (leftWheel->state == 1) {
        if (leftWheel->prestate == 4) {
            leftWheel->direction = true;
        } else if (leftWheel->prestate != 1) {
            leftwheel->direction = false;
        }
    } else if (leftWheel->state == 4) {
        if (leftWheel->prestate == 1) {
            leftWheel->direction = false;
        } else if (leftWheel->prestate != 4) {
            leftWheel->direction = true;
        }
    } else {
        leftWheel->direction = (leftWheel->state > leftWheel->prestate);
    }

    /* TODO ..do nothing if state ain't change */
}
