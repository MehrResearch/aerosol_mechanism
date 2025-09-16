from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 100
n = 50
reaction_delay = 5
cycle_delay = 10
off_time = 0

ctrlaer = CtrlAer(sm_number=0, base_pin=13, n_pins=2,freq=113_500)

air_control = Pin(0, Pin.OUT)
air_control.value(0)
sleep(2)

def prog():
    for i in range(N):
        for j in range(n):
            yield ON, 50
            yield OFF, 50
        sleep(5)

# GP14:  aldehyde + acid
# GP13:  amine
ctrlaer.run(prog())
