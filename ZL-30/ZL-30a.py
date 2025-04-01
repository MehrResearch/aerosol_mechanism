from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 3
n = 30
reaction_delay = 5
cycle_delay = 120
off_time = 250

ctrlaer = CtrlAer(sm_number=0, base_pin=7, n_pins=2,freq=111_500)

air_control = Pin(16, Pin.OUT)
air_control.value(0)
sleep(2)

def prog():
    for i in range(N):
        for cmd in [0b01,0b10,0b11]:
            for j in range(n):
                print(f'{j}: amine: {cmd % 2} glue: {cmd // 2}')
                yield cmd, 50
                yield OFF, off_time
            ctrlaer.block()
            sleep(reaction_delay)
            air_control.value(1)
            sleep(cycle_delay)
            air_control.value(0)
            
# GP07:  amine
# GP08:  glue
#ctrlaer.run(prog())