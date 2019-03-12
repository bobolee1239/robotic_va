/* file: pwm.cpp */

#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static const int PWM_pin = 1;  /* GPIO 1 for wiringPi = GPIO 18 for BCM */

int main(int argc, char* argv[])
{
    int intensity;

    if(wiringPiSetup() == -1){
        exit(-1);
    }

    /* set pwm pin as output */
    pinMode(PWM_pin, PWM_OUTPUT);

    while(1){
        fprintf(stdout, "half speed ...\n");
        pwmWrite(PWM_pin, 512);
        sleep(3);

        fprintf(stdout, "stop for awhile ...\n");
        pwmWrite(PWM_pin, 0);
        sleep(3);
    }

    return 0;
}



