import RPi.GPIO as GPIO
from time import sleep

GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(13, GPIO.HIGH)
sleep(1)
GPIO.output(13, GPIO.LOW)
sleep(1)
GPIO.cleanup()