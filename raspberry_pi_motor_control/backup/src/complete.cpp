// Copyright 2019 Tsung-Han Lee
/**************************************
 ** FILE    : complete.cpp
 ** AUTHOR  : Tsung-Han Lee
 **************************************/

#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>

#include "hall_sensor_decode.h"

/********************* DEFINITION ***************************************/
int initHallSensors();
int initPWM();
int initTimerISR();
void timerISR(int signum);

typedef volatile struct PIController {
    volatile double rpm;
    volatile double piOut;      //   output of PI controller
    volatile double err;       //    error
    volatile double ierr;     //     integration of error
    double kp;
    double ki;
} PIController_t;

/***********************************************************************/

/********************* PARAMETERS ***************************************/
const int leftPWM   = 1;      // GPIO 1 for wiringPi = GPIO 18 for BCM
const int rightPWM  = 23;    // GPIO 23 for wiringPi = GPIO 13 for BCM
const int rightPWMn = 24;   // GPIO 24 for wiringPi  = GPIO 19 for BCM
const int leftPWMn  = 26;  // GPIO 26 for wiringPi   = GPIO 12 for BCM

PIController_t leftController  = {0.0, 0.0, 0.0, 0.0, 0.008, 0.02};
PIController_t rightController = {0.0, 0.0, 0.0, 0.0, 0.008, 0.02};

double leftRef  = 25.0;
double rightRef = 25.0;
double Ts       = 0.01;       // sampling interval

/***********************************************************************/

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

    initPWM();
    initTimerISR();

    while (1) {
        /* output sensor */
        /*
        std::cout << "l:" << std::fixed << std::setprecision(2)
                  << leftController.rpm << " , ";
        std::cout << "r:" << std::fixed << std::setprecision(2)
                  << rightController.rpm << std::endl;
        */
        sleep(1);
    }

    /* release memory */
    closeHallSensor();
    return 0;
}

void timerISR(int signum) {
    /*********** MOTOR1 **********/
    // measure TODO... translate rpm correctly : gear ratio
    leftController.rpm = leftWheel->numStateChange * 1.04166;
    // DEBUG
    std::cout << std::fixed << std::setprecision(2)
              << leftController.rpm << std::endl;
    std::cout << "#change: " << leftWheel->numStateChange << std::endl;

    leftWheel->numStateChange = 0;

    /**************** PI Controller *******************/
    leftController.err = leftRef - leftController.rpm;
    leftController.ierr += Ts*leftController.err;

    // limit integration output
    if (leftController.ierr > 50.0) {
        leftController.ierr = 50.0;
    } else if (leftController.ierr < -50.0) {
        leftController.ierr = -50.0;
    }
    leftController.piOut = leftController.kp * leftController.err
                           + leftController.ki * leftController.ierr;
    // saturation
    if (leftController.piOut > 0.5) {
        leftController.piOut = 0.5;
    } else if (leftController.piOut < -0.5) {
        leftController.piOut = -0.5;
    }
    // output to acuator : complement PWM
    pwmWrite(leftPWM, static_cast<int>(1024*(0.5 + leftController.piOut)));
    /**************************************************/
    /*********** MOTOR2 **********/
    // measure TODO... translate rpm correctly : gear ratio
    rightController.rpm = rightWheel->numStateChange * 1.04166;
    rightWheel->numStateChange = 0;

    /**************** PI Controller *******************/
    rightController.err = rightRef - rightController.rpm;
    rightController.ierr += Ts*rightController.err;

    // limit integration output
    if (rightController.ierr > 50.0) {
        rightController.ierr = 50.0;
    } else if (rightController.ierr < -50.0) {
        rightController.ierr = -50.0;
    }
    rightController.piOut = rightController.kp * rightController.err
                           + rightController.ki * rightController.ierr;
    // saturation
    if (rightController.piOut > 0.5) {
        rightController.piOut = 0.5;
    } else if (rightController.piOut < -0.5) {
        rightController.piOut = -0.5;
    }
    // output to acuator : complement PWM
    pwmWrite(rightPWM, static_cast<int>(1024*(0.5 + rightController.piOut)));
    /**************************************************/
}

int initTimerISR() {
    struct sigaction sa;
    struct itimerval timer;

    /* Install timer_handler as the signal handler for SIGALRM (Real timer) */
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = &timerISR;
    sigaction(SIGALRM, &sa, NULL);

    /* Configure the timer to be expired after 10 ms */
    timer.it_value.tv_sec  = 0;
    timer.it_value.tv_usec = 10000;
    /* Configure the timer back to 10 ms after expired */
    timer.it_interval.tv_sec  = 0;
    timer.it_interval.tv_usec = 10000;

    /* Start a real timer */
    setitimer(ITIMER_REAL, &timer, NULL);

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

    return 0;
}

void hallSensor_ISR(void) {
    /* Dealing with left motor */
    /* read state of GPIO pin */
    leftWheel->hallA = digitalRead(LEFT_HALL_A);
    leftWheel->hallB = digitalRead(LEFT_HALL_B);

    /***********************************************************
     **   I. hallA hallB should be 1, 0 => bitwise xor is fine
     **  II. update state and check if:
     **        1. state increase (direction = forward)
     **        2. state decrease (direction = backward)
     **        3. same state     (nothing happened)
     **
     **  [HALL SENSOR TABLE] :
     **         * FORWARD   : from up to down
     **         * BACKWARD  : from down to up
     **   ---------------------------
     **   | STATE | HALL A | HALL B |
     **   |   1   |   0    |   0    |
     **   |   2   |   0    |   1    |
     **   |   3   |   1    |   1    |
     **   |   4   |   1    |   0    |
     **   |   1   |   0    |   0    |
     **   ---------------------------
     ***********************************************************/
    // update state in [1, 2, 3, 4]
    leftWheel->state = (leftWheel->hallA << 1)
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
            ++(leftWheel->numStateChange);
        } else {
            --(leftWheel->numStateChange);
        }
        //  DEBUG
        //  std::cout << leftWheel->prestate << " -> " << leftWheel->state << std::endl;
        std::cout << leftWheel->numStateChange << std::endl;
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

void closeHallSensor() {
    delete rightWheel;
    delete leftWheel;
}
