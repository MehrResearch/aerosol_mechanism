from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 2
n = 30
reaction_delay = 120
cycle_delay = 240
off_time = 500

ctrlaer = CtrlAer(sm_number=0, base_pin=8, n_pins=3,freq=114_500)

air_control = Pin(16, Pin.OUT)
air_control.value(0)
print("value_off")
sleep(2)

def prog():
    for i in range(N):
        if i % 2 == 0:
            # even: spray not at the same time
            for j in range(n):
                print(f'{j}:  TFB')
                yield 0b001, 50
                yield OFF, off_time//2
                print(f'{j}:  p-phenylenediamine')
                yield 0b100, 50
                yield OFF, off_time//2               
        else:
            for j in range(n):
                print(f'{j}:  Both')
                yield 0b101, 50
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
