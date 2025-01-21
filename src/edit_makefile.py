#!/usr/bin/env python3

import optparse
import os

def process_continuation_lines(lines):
    longline = ''
    for line in lines:
        longline += line
    longline = longline.replace('\\\n','')
    ret = longline.split('\n')
    for idx,r in enumerate(ret):
        ret[idx] = r+'\n'
    return ret

def gen_newmake(machine, basename, orgMakefile):
    fmake=open(orgMakefile)
    LAUNCHER='$(LAUNCHER)'
    newMake = ''
    command = ''
    foundLAUNCHER = False
    lines = process_continuation_lines(fmake.readlines())
    fmake.close()
    skip_ids=[]
    for idx, line_bare in enumerate(lines):
        if idx in skip_ids:
            continue
        line = line_bare.strip()
        if line.startswith('LAUNCHER'):
            words = line.split('=')
            if len(words)==1 or len(words[1])==0:
                line = 'LAUNCHER = bash -c'
            newMake += line+'\n'
            foundLAUNCHER = True
        elif line.startswith('run:'):
            depar = line.split('run:')
            dep = ''
            if len(depar)>1:
                dep = depar[1].strip()
            nextLine = lines[idx+1]
            nextWords = nextLine.split(LAUNCHER)
            if len(nextWords)<2:
                print('invalid COMMAND line : '+nextLine)
                return None, None
            newMake += 'include ../'+basename+'-'+machine+'-cmd\n'
            newMake += '\n'
            newMake += line+'\n'
            newMake += '\t'+LAUNCHER+' "$(COMMAND)"\n'
            il = 2
            skip_ids.append(idx+1)
            # process lines which start with a tab character
            while(True):
                if len(lines)<=idx+il:
                    break
                nline = lines[idx+il]
                if nline is None or len(nline)==0 or not nline.startswith('\t'):
                    break
                #newMake += nline
                skip_ids.append(idx+il)
                il += 1

            baseCommand = nextLine.split(LAUNCHER)[1]
            #print('baseCommand '+baseCommand)
            command  = 'COMMAND = '+baseCommand+'\n'
            newMake += '\n'
            newMake += 'run_time: ' + dep+'\n'
            newMake += '\t'+LAUNCHER+' "time $(COMMAND)"'+'\n'
            newMake += '\n'
            newMake += 'run_mem: '  + dep+'\n'
            if machine=='NVIDIA':
                newMake += '\t'+'nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop-ms=1 > log.mem &\n'
                newMake += '\t'+LAUNCHER+' "time $(COMMAND)"\n'
                newMake += '\t'+'pkill nvidia-smi\n'
                newMake += '\t'+'../m.sh\n'
            elif machine=='AMD':
                newMake += '\t'+'../rocm-smi.sh > log.mem &\n'
                newMake += '\t'+LAUNCHER+' "time $(COMMAND)"\n'
                newMake += '\t'+'pkill rocm-smi.sh\n'
                newMake += '\t'+'../rocm-m.sh\n'
        else:
            newMake += line_bare
    if not foundLAUNCHER:
        newMake = 'LAUNCHER = bash -c\n'+newMake
    return newMake, command

if __name__ == '__main__':
    parser = optparse.OptionParser(usage="%prog [options]",description="edit Makefiles for the HeCBenchmark")
    parser.add_option('-t','--target',type='choice',choices=('NVIDIA','AMD'),dest='target',default='NVIDIA')
    parser.add_option('-o','--output',type=str,dest='output',default=None)
    parser.add_option('-i','--input', type=str,dest='input', default='Makefile')
    (options, args) = parser.parse_args()
    output = options.output
    if output is None:
        if options.target == 'NVIDIA':
            output = 'Makefile.NVD'
        if options.target == 'AMD':
            output = 'Makefile.AMD'
    basename = '-'.join(os.path.basename(os.getcwd()).split('-')[:-1])
    foutput = open(output,'w')
    makestr, command = gen_newmake(options.target, basename=basename, orgMakefile=options.input)
    foutput.write(makestr)
    foutput.close()
    fcmd = open('../'+basename+'-'+options.target+'-cmd', 'w')
    fcmd.write(command)
    fcmd.close()

