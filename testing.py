from time import sleep

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT, initial=GPIO.LOW)
try:
    while True:
        GPIO.output(16, GPIO.HIGH)
        sleep(1)
        GPIO.output(16, GPIO.LOW)
        sleep(1)
finally:
    GPIO.cleanup()
