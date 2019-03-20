// Copyright 2019 Tsung-Han Lee
/**************************************
 ** FILE    : motorControl.cpp
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


    double d1, d2;
    std::cout << "input d1, d2: ";
    std::cin >> d1 >> d2;
    pwmWrite(leftPWM, static_cast<int>(1024*d1));
    pwmWrite(rightPWM, static_cast<int>(1024*d2));

    initTimerISR();

    /* output sensor */
    
    while (1) {
        sleep(1);
    }

    initPWM();
    /* release memory */
    closeHallSensor();
    return 0;
}

void timerISR(int signum) {
    /*********** MOTOR1 **********/
    // measure TODO... translate rpm correctly : gear ratio
    leftController.rpm = leftWheel->numStateChange * 1.04166;
    std::cout << "l:" << std::fixed << std::setprecision(2)
                  << leftController.rpm << std::endl;
    leftWheel->numStateChange = 0;

    /**************** PI Controller *******************/
    leftController.err = leftRef - leftController.rpm;
    leftController.ierr += Ts*leftController.err;

    /*********** MOTOR2 **********/
    // measure TODO... translate rpm correctly : gear ratio
    rightController.rpm = rightWheel->numStateChange * 1.04166;
    rightWheel->numStateChange = 0;

    /**************** PI Controller *******************/
    rightController.err = rightRef - rightController.rpm;
    rightController.ierr += Ts*rightController.err;
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
