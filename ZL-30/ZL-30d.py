from pico_ctrlaer import ON, OFF, mux, CtrlAer
from machine import ADC, Pin
from time import sleep

N = 10
n = 10
reaction_delay = 15
cycle_delay = 180
off_time = 250
on_time = 50

ctrlaer = CtrlAer(sm_number=0, base_pin=7, n_pins=2,freq=111_500)

air_control = Pin(16, Pin.OUT)
air_control.value(0)
sleep(2)

def Glue():
    for i in range(N):
        Glue_time= int(on_time)
        for j in range(n):
            print(f'{(i, j)}: Glue {Glue_time}')
            yield ON, on_time
            yield OFF, off_time
        ctrlaer.block()
        sleep(reaction_delay)
        air_control.value(1)
        sleep(cycle_delay)
        air_control.value(0)

def morphline():
    for i in range(N):
        morphline_time = int(on_time * i / (N - 1))
        for j in range(n):
            print(f'{(i, j)}: morphline {morphline_time}')
            yield OFF, on_time - morphline_time
            yield ON, morphline_time
            yield OFF, off_time


prog = mux([morphline(), Glue()])
            
# GP07:  morphline
# GP08:  Glue
ctrlaer.run(prog)