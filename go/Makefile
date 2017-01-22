CC=gcc 
CFLAGS=-Wall

all: go fo
go: go.o gpio_lib.o
go.o: go.c gpio_lib.h

fo: fo.o gpio_lib.o
fo.o: fo.c gpio_lib.h

gpio_lib.o: gpio_lib.c gpio_lib.h

