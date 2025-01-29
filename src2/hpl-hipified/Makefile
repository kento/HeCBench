LAUNCHER =

program = main

$(program):
	cd src/hpl-2.3/; make 

clean:
	cd src/hpl-2.3/; make clean

run: $(program)
     $(LAUNCHER) mpirun -n 1 ./src/hpl-2.3/bin/intel64/xhpl
     
