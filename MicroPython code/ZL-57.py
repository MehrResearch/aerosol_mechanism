from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 1
ctrlaer = CtrlAer(sm_number=0, base_pin=7, n_pins=4,freq=114_500)
air_control = Pin(16, Pin.OUT)
air_control.value(1)
sleep(2)

def prog():
    for i in range(N):
        for q in range(100):
            for pin in [0b0101] + [0]*3:
                print(f'{i} {q}: test-test')
                yield pin, 10
                #yield OFF, 100       

        
# MS Heater off
# GP10:  A
# GP11:  B
# GP12:  C
ctrlaer.run(prog())
