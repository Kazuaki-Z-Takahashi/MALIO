
### Import main tools. ###
import MDAnalysis
import os
import glob
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i","--input",
    default = "test-s",
    help = "specify input gro/xtc prefix")
parser.add_argument(
    "-o","--output",
    default = "structure",
    help = "specify output directory prefix")
parser.add_argument(
    "-n","--number_of_structures",
    type = int,
    default = 0,
    help = "specify number of structures")

args = parser.parse_args()
input  = args.input
output = args.output
nstruct = args.number_of_structures

if nstruct < 1:
    parser.print_help()
    sys.exit(1)

for istruct in range(nstruct):
    
    fname = "__input/" + input  + str(istruct + 1)
    sname = "__input/" + output + str(istruct + 1) + "/"

    os.makedirs(sname, exist_ok=True)

    for f in glob.glob(sname + '*.txt'):
        if os.path.isfile(f):
            os.remove(f)

    u = MDAnalysis.Universe(fname + ".gro", fname + ".xtc", convert_units=False)
    
    for n, ts in enumerate(u.trajectory):
        
        ny = u.select_atoms("name NY1")
        ca = u.select_atoms("name CA12")
        nmol = len(ny)
        rny = ny.positions
        rca = ca.positions
        lx = ts.dimensions[0]
        ly = ts.dimensions[1]
        lz = ts.dimensions[2]
        coord = (rny + rca) * 0.5
        direct = rny - rca

        lines = []
        lines.append('{:.16e} {:.16e} {:.16e}\n'.format(lx, ly, lz))
        lines.append(str(nmol) + "\n")
        for i in range(nmol):
            line = " ".join(['{:.16e}'.format(d) for d in (*coord[i], *direct[i])]) + "\n"
            lines.append(line)

        f = open(sname + format(n+1, '#04') + ".txt", "w")
        f.writelines(lines)
        f.close()

