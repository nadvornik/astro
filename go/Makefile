CC=gcc 
CFLAGS=-Wall

all: go fo temp_sensor ext_trigger
go: go.o gpio_lib.o
go.o: go.c gpio_lib.h

fo: fo.o gpio_lib.o
fo.o: fo.c gpio_lib.h

temp_sensor: temp_sensor.o gpio_lib.o
temp_sensor.o: temp_sensor.c gpio_lib.h

ext_trigger: ext_trigger.o gpio_lib.o
ext_trigger.o: ext_trigger.c gpio_lib.h

gpio_lib.o: gpio_lib.c gpio_lib.h

