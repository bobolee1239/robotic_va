//  Copyright (c) 2019 Tsung-Han Lee
/**************************************
 ** FILE    : mobileBot.cpp
 ** AUTHOR  : Tsung-Han Lee
 **************************************/

#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "hall_sensor_decode.h"

#define R   0.05    //  Unit: meter
#define L   0.27    //  Unit: meter

/********************* DEFINITION ***************************************/
int initHallSensors();
int initPWM();
int initTimerISR();
void timerISR(int signum);

typedef volatile struct PIController {
    volatile double rpm;
    volatile double piOut;      //  output of PI controller
    volatile double err;        //  error
    volatile double ierr;       //  integration of error
    double kp;
    double ki;
} PIController_t;

/***********************************************************************/

/********************* PARAMETERS ***************************************/
const int leftPWM   = 24;     //  WiringPi  1 : BCM 18
const int leftPWMn  = 23;     //  WiringPi 26 : BCM 12
const int rightPWM  = 1;      //  WiringPi 23 : BCM 13
const int rightPWMn = 26;     //  WiringPi 24 : BCM 19

PIController_t leftController  = {0.0, 0.0, 0.0, 0.0, 0.008, 0.02};
PIController_t rightController = {0.0, 0.0, 0.0, 0.0, 0.008, 0.02};

const double Ts     = 0.01;       //  sampling interval
double leftRef      = 0.0;        //  reference signal for left wheel
double rightRef     = 0.0;        //  reference signal for right wheel
double angle2Rotate = 0.0;        //  angle to rotate for mobile robot
/***********************************************************************/

int main(int argc, char* argv[]) {
    if (argc != 2) {
      std::cout << "[Usage] ./mobileBot <rotation_angle>" << std::endl;
      closeHallSensor();
      return -1;
    }
    /* Setup wiringPi */
    if (wiringPiSetup() < 0) {
        std::cerr << "Unable to setup wiringPi: "
                  << strerror(errno) << std::endl;
        closeHallSensor();
        return -1;
    }

    if (initHallSensors() < 0) {
        std::cerr << "Init hall sensors failed!" << std::endl;
        closeHallSensor();
        return -1;
    }

    initPWM();
    initTimerISR();

    angle2Rotate = static_cast<double>(atoi(argv[1]));
    while (1) {
        /* receive rotation angle and set timeout */
        /* output sensor information for debug sake */
#ifdef DEBUG
        std::cout << "l:" << std::fixed << std::setprecision(2)
                  << leftController.rpm << " , ";
        std::cout << "r:" << std::fixed << std::setprecision(2)
                  << rightController.rpm << std::endl;
#endif
        sleep(1);
    }

    /* release memory */
    closeHallSensor();
    return 0;
}

void timerISR(int signum) {
    /**
     **  Update rotation history : 6 for the coef from rpm -> degree
     **/
    angle2Rotate -= ((rightRef - leftRef) * 0.01 * R / L * 6);
    //  give command
    if (std::abs(angle2Rotate) < 0.001) {
      leftRef  = 0.0;
      rightRef = 0.0;
    } else if (angle2Rotate > 0) {
      leftRef  = -25.0;
      rightRef = 25.0;
    } else {
      leftRef  = 25.0;
      rightRef = -25.0;
    }
    /*********** MOTOR1 **********/
    // measure TODO... translate rpm correctly : gear ratio
    leftController.rpm = leftWheel->numStateChange * 1.04166;
    leftWheel->numStateChange = 0;

    /**************** PI Controller *******************/
    leftController.err = leftRef - leftController.rpm;
    leftController.ierr += Ts*leftController.err;

    std::cout << "err: " << leftController.err << std::endl;
    std::cout << "kp: " << leftController.kp
              << ", ki: " << leftController.ki << std::endl;

    // limit integration output
    if (leftController.ierr > 50.0) {
        leftController.ierr = 50.0;
    } else if (leftController.ierr < -50.0) {
        leftController.ierr = -50.0;
    }
    leftController.piOut = leftController.kp * leftController.err
                           + leftController.ki * leftController.ierr;
    leftController.piOut *= -1;
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
