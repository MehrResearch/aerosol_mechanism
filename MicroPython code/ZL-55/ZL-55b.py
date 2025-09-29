from ctrlaer import ON, OFF, mux, CtrlAer
from machine import Pin
from time import sleep

n = 10
reaction_delay = 5
cycle_delay = 120
off_time = 200
on_time = 100

ctrlaer = CtrlAer(sm_number=0, base_pin=7, n_pins=3,freq=110_500)

air_control = Pin(16, Pin.OUT)
air_control.value(0)
sleep(2)

Ratios = [0, 25, 50, 75, 100, 100, 75, 50, 25, 0]
Glue_time = [int(on_time * (100 - i) / 100) for i in Ratios]
Morpholine_time = [int(on_time * i / 100) for i in Ratios]
print(Glue_time, Morpholine_time)

def Glue():
    for t in Glue_time:
        for j in range(n):
            print(f'Glue = {t}')
            yield ON, t
            yield OFF, off_time + on_time - t
        ctrlaer.block()
        sleep(reaction_delay)
        air_control.value(1)
        sleep(cycle_delay)
        air_control.value(0)
        
def morphline():
    for t in Morpholine_time:
        for j in range(n):
            print(f'morpholine = {t}')
            yield ON, t
            yield OFF, off_time + on_time - t
            
def inactive():
    while True:
        yield OFF, 100000

def observe(c):
    for cmd, t in c:
        if cmd:
            print(f'{cmd:03b}: {t}')
        yield cmd, t


# GP10:  Glue
# GP12:  Morphline
for _ in range(20):
    prog = mux([Glue(), inactive(), morphline()])
    ctrlaer.run(observe(prog))

air_control.value(0)