from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 1
n = 5
reaction_delay = 10
cycle_delay = 180
off_time = 250

ctrlaer = CtrlAer(sm_number=0, base_pin=7, n_pins=3,freq=115_500)

air_control = Pin(16, Pin.OUT)
air_control.value(0)
sleep(10)

def prog():
    for i in range(N):
        for cmd in [0b001, 0b100, 0b101]:
            for j in range(n):
                print(f'{j}: amine: {cmd % 2} aldehyde: {cmd // 2}')
                yield cmd, 50
                yield OFF, off_time
            ctrlaer.block()
            sleep(reaction_delay)
            air_control.value(1)
            sleep(cycle_delay)
            air_control.value(0)
            sleep(10)
# GP7:  L-serine
#ctrlaer.run(prog())
air_control.value(1)
