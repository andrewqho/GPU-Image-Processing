CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp line_detection.cpp
PROG = main

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

all: $(PROG)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean: 
	rm $(PROG)