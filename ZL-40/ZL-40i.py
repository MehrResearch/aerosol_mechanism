from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 3
n = 30
reaction_delay = 120
cycle_delay = 240
off_time = 2000

ctrlaer = CtrlAer(sm_number=0, base_pin=8, n_pins=3,freq=114_500)

air_control = Pin(16, Pin.OUT)
air_control.value(1)

def prog():
    for j in range(1000):
        print(f'{j}:  Both')
        yield 0b101, 50
        yield OFF, off_time
# MS Heater off           
# GP08:  TFB
# GP10:  pamine
ctrlaer.run(prog())
