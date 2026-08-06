from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 10
reaction_delay = 15
cycle_delay = 120
off_time = 250

ctrlaer = CtrlAer(sm_number=0, base_pin=7, n_pins=3,freq=115_500)

air_control = Pin(16, Pin.OUT)
#out_control = Pin(?, Pin.OUT)
air_control.value(0)
sleep(2)

def prog():
    for i in range(N):
        for times, cmd in [(5,0b001), (5, 0b100), (5, 0b101)]:
            for j in range(times):
                print(f'{j}: amine: {cmd % 2} aldehyde: {cmd // 2}')
                yield cmd, 50
                yield OFF, off_time
            ctrlaer.block()
            sleep(reaction_delay)
            air_control.value(1)
            sleep(cycle_delay)
            air_control.value(0)
            
# GP14:  aldehyde
# GP13:  amine
ctrlaer.run(prog())
