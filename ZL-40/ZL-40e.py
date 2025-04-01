from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 2
n = 50
reaction_delay = 120
cycle_delay = 180
off_time = 250

ctrlaer = CtrlAer(sm_number=0, base_pin=8, n_pins=3,freq=114_500)

air_control = Pin(16, Pin.OUT)
air_control.value(0)
print("value_off")
sleep(2)

def prog():
    for i in range(N):
        for cmd in [0b001, 0b100, 0b101]:
            for j in range(n):
                print(f'{j}:  TFB: {cmd % 2} pamine: {cmd // 2}')
                yield cmd, 50
                yield OFF, off_time
            ctrlaer.block()
            sleep(reaction_delay)
            print("reaction_delay")
            air_control.value(1)
            print("value_on")
            sleep(cycle_delay)
            print("cycle_delay")
            air_control.value(0)
            print("value_off")
# MS Heater off           
# GP08:  TFB
# GP10:  pamine
ctrlaer.run(prog())