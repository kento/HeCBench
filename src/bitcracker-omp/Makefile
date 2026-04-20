#===============================================================================
# User Options
#===============================================================================

CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp w_blocks.cpp attack.cpp utils.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp bitcracker.h sha256.h aes.h
$(CC) $(CFLAGS) -c $< -o $@

clean:
rm -rf $(program) $(obj)

run: $(program)
$(LAUNCHER) ./$(program) -f hash_pass/img_win8_user_hash.txt -d hash_pass/user_passwords.txt
