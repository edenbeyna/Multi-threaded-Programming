CC=g++
CXX=g++
RANLIB=ranlib

LIBSRC=MapReduceFramework.cpp MapReduceFramework.h Barrier.cpp Barrier.h
LIBOBJ=$(LIBSRC:.cpp=.o)
LIBCLEAN=libMapReduceFramework.a Barrier.o MapReduceFramework.o ex3.tar
LIBTAR=MapReduceFramework.cpp Barrier.cpp Barrier.h

INCS=-I.
CFLAGS = -Wall -pthread -std=c++11 -g $(INCS)
CXXFLAGS = -Wall -pthread -std=c++11 -g $(INCS)

MAPREDUCEFRAMEWORKLIB = libMapReduceFramework.a
TARGETS = $(MAPREDUCEFRAMEWORKLIB)

TAR=tar
TARFLAGS=-cvf
TARNAME=ex3.tar
TARSRCS=$(LIBTAR) Makefile README

all: $(TARGETS)

$(TARGETS): $(LIBOBJ)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@

clean:
	$(RM) $(TARGETS) $(OBJ) $(LIBCLEAN) *~ *core

depend:
	makedepend -- $(CFLAGS) -- $(SRC) $(LIBSRC)

tar:
	$(TAR) $(TARFLAGS) $(TARNAME) $(TARSRCS)
