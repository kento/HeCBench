#!/usr/bin/env python3

import optparse
import os
import sys
from matplotlib import pyplot as plt

def extend_markdown(markd):
    f = open(markd)
    firstline = f.readline().strip()
    firstline += ' plot |'
    print(firstline)
    secondline = f.readline().strip()
    secondline += ' -- |'
    print(secondline)
    for line in f:
        line = line.strip()
        bname = line.split('|')[1].strip()
        line += '!['+bname+'](SVGs/'+bname+'.svg) |'
        print(line)
    f.close()

def gen_plt(x, line):
    words = line.strip().split('|')
    bname = words[1].strip()
    words = words[2:len(words)-1]
    y = []
    for w in words:
        w = w.strip()
        try:
            fl = float(w)
        except ValueError:
            fl = 0.0
        y.append(fl)
    plt.close()
    fig, ax = plt.subplots()
    ax.set_ylabel('time (s)')
    plt.bar(x, y)

    outf = 'SVGs/'+bname+'.svg'
    plt.savefig(outf)

if __name__ == '__main__':
    parser = optparse.OptionParser(usage="%prog [options]",description="generate bar graph from the markdown table")
    parser.add_option('-m', '--markdown', dest='markdown', default=None)
    parser.add_option('-e', '--extend', dest='extend', action='store_true', default=False)
    parser.add_option('-n', '--noplot', dest='noplot', action='store_false', default=True)

    (options, args) = parser.parse_args()

    if options.markdown is None or not os.path.exists(options.markdown):
        print('the specified markdown file does not exist')
        sys.exit()
    f = open(options.markdown)
    line0 = f.readline()
    columns = line0.strip().split('|')
    x = columns[2:len(columns)-1]
    dumm = f.readline()

    if not options.noplot:
        for line in f:
            line = line.strip()
            gen_plt(x,line)
    f.close()

    if options.extend:
        extend_markdown(options.markdown)

