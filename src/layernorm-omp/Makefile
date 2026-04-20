CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

program = main
source = main.cpp
obj = $(source:.cpp=.o)

CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall -I../layernorm-cuda
LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif

$(program): $(obj)
$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
$(CC) $(CFLAGS) -c $< -o $@

clean:
rm -rf $(program) $(obj)

run: $(program)
$(LAUNCHER) ./$(program) 
