// Copyright 2019 Tsung-Han Lee
/**************************************
 ** FILE    : hall_sensor_decode.cpp
 ** AUTHOR  : Tsung-Han Lee
 **************************************/
#include "hall_sensor_decode.h"

WheelState_t* const leftWheel = new WheelState_t();
WheelState_t* const rightWheel = new WheelState_t();

memset((void*)leftWheel, 0, sizof(WheelState_t));
memset((void*)rightWheel, 0, sizeof(WheelState_t));

void hallSensor_ISR(void) {
    /* Dealing with left motor */
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
            leftWheel->direction = false;
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

    /* do nothing if state ain't change */
    if (leftWheel->prestate != leftWheel->state) {
        if (leftWheel->direction) {
            ++leftWheel->numStateChange;
        } else {
            --leftWheel->numStateChange;
        }
    }

    /* update previous state */
    leftWheel->prestate = leftWheel->state;

    /* Dealing with right motor */
    /* read state of GPIO pin */
    rightWheel->hallA = digitalRead(RIGHT_HALL_A);
    rightWheel->hallB = digitalRead(RIGHT_HALL_B);

    /* hallA hallB should be 1, 0 */
    rightWheel->state = (rightWheel->hallA << 2)
                        + (rightWheel->hallA ^ rightWheel->hallB)
                        + 1;

    if (rightWheel->state == 1) {
        if (rightWheel->prestate == 4) {
            rightWheel->direction = true;
        } else if (rightWheel->prestate != 1) {
            rightWheel->direction = false;
        }
    } else if (rightWheel->state == 4) {
        if (rightWheel->prestate == 1) {
            rightWheel->direction = false;
        } else if (rightWheel->prestate != 4) {
            rightWheel->direction = true;
        }
    } else {
        rightWheel->direction = (rightWheel->state > rightWheel->prestate);
    }

    /* do nothing if state ain't change */
    if (rightWheel->prestate != rightWheel->state) {
        if (rightWheel->direction) {
            ++rightWheel->numStateChange;
        } else {
            --rightWheel->numStateChange;
        }
    }

    /* update previous state */
    rightWheel->prestate = rightWheel->state;
}
