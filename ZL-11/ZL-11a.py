from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 10
n = 50
reaction_delay = 5
cycle_delay = 180
off_time = 250

ctrlaer = CtrlAer(sm_number=0, base_pin=13, n_pins=2,freq=113_500)

air_control = Pin(0, Pin.OUT)
air_control.value(0)
sleep(2)

def prog():
    for i in range(N):
        for cmd in [0b01]:
            for j in range(n):
                print(f'{j}: amine+aldehyde: {cmd % 2} MeOH: {cmd // 2}')
                yield cmd, 50
                yield OFF, off_time
            ctrlaer.block()
            sleep(reaction_delay)
            air_control.value(1)
            sleep(cycle_delay)
            air_control.value(0)
# MS Heater off           
# GP14:  MeOH
# GP13:  amine+aldehyde
ctrlaer.run(prog())

