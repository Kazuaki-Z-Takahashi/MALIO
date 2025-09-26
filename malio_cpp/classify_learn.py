### Import MALIO ###
from mpi4py import MPI
import argparse
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "-fi","--filenumber",
    nargs=3,
    default = [-1, -1, -1],
    help = "specify file name of begining, number and interval")
parser.add_argument(
    "-j","--joblib",
    default = "",
    help = "specify joblib file name")
parser.add_argument(
    "-df","--data_frame",
    default = "data_frame",
    help = "specify data frame file prefix")
args = parser.parse_args()
filenumber = args.filenumber
joblib_file = args.joblib
data_frame_prefix = args.data_frame

bnum = int(filenumber[0])
snum = int(filenumber[1])
dnum = int(filenumber[2])

if joblib_file == "" or bnum < 0 or snum < 0 or dnum < 0:
  print(parser.usage())
  sys.exit()
  

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


### Import main tools ###
import numpy as np
import joblib
#from sklearn.externals import joblib


### Import for signal handling ###
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)


### Set number of threads/cores of your computer ###
if (rank == 0):
  print ("MPI processes = {0}".format(size))


### Compute OP for big trajectory ###
# Make list of file names to be readed/written #
dlist = [0]
#dlist = [0,5,10,15,20,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]

if (rank ==0):
  print("For debug, dnum, len(dlist):", dnum, len(dlist))

filename_list_r = []
filename_list_w = []
filename_list_p = []
for i in dlist:
  for j in range(snum):
    filename_list_r.append('__input/'  + str(i + bnum + j*dnum).zfill(7)+'.txt')
    filename_list_w.append('__output/' + str(i + bnum + j*dnum).zfill(7)+'.txt')
    filename_list_p.append(str(i + bnum + j*dnum).zfill(7))

if(rank == 0):
  print("Total number of processing SS    :", len(filename_list_r))
  os.makedirs("__output", exist_ok=True)
  
# Loop for LOPs calculation start #
import time
initial_time = time.time()
t1 = time.time()
n = 0
for file_name in filename_list_r:

  # Write elapsed time. #
  if (n % 10) == 1:
    t2 = time.time() - t1
    if (rank == 0):
      print ("### Record ### Comput.OP:{0}".format(n) + "[times]")
      print ("### Record ### Elap.Time:{0}".format(t2) + "[sec]")

# Read coordinates #
  if (rank == 0):
    file_data = open(filename_list_r[n], 'r')
    lines = file_data.readlines()
    file_data.close()
    mid  = []
    posx = []
    posy = []
    posz = []
    qua1 = []
    qua2 = []
    qua3 = []
    qua4 = []
    iid = -1
    ix = -1
    iy = -1
    iz = -1
    iq1 = -1
    iq2 = -1
    iq3 = -1
    iq4 = -1
    bQuat = False
    for i, line in enumerate(lines):
      l = line.strip()
      words = l.split()
      if l == "ITEM: NUMBER OF ATOMS":
        nmol = int(lines[i+1].strip())
        continue
      if l[:16] == "ITEM: BOX BOUNDS":
        lx = float(lines[i+1].split()[1])
        ly = float(lines[i+2].split()[1])
        lz = float(lines[i+3].split()[1])
        continue
      if l[:11] == "ITEM: ATOMS":
        for j in range(2, len(words)):
          if words[j] == "id"         : iid = j - 2
          if words[j] == "x"          : ix  = j - 2
          if words[j] == "y"          : iy  = j - 2
          if words[j] == "z"          : iz  = j - 2
          if words[j] == "c_orient[1]": iq1 = j - 2
          if words[j] == "c_orient[2]": iq2 = j - 2
          if words[j] == "c_orient[3]": iq3 = j - 2
          if words[j] == "c_orient[4]": iq4 = j - 2
        if iid < 0 or ix < 0 or iy < 0 or iz < 0:
          break
        bQuat = (iq1 > -1 and iq2 > -1 and iq3 > -1 and iq4 > -1)
        for j in range(nmol):
          line = lines[i+1+j]
          words = line.split()
          mid.append(int(words[iid]))
          posx.append(float(words[ix]))
          posy.append(float(words[iy]))
          posz.append(float(words[iz]))
          if bQuat: qua1.append(float(words[iq1]))
          if bQuat: qua2.append(float(words[iq2]))
          if bQuat: qua3.append(float(words[iq3]))
          if bQuat: qua4.append(float(words[iq4]))

  ### if (rank == 0):
    print(nmol)
    print(lx,ly,lz)
    print(posx[0])
    print(posx[nmol-1])
    print(qua1[0])
    print(qua1[nmol-1])
  else:
    nmol = None
    lx = None
    ly = None
    lz = None
    mid = None
    posx = None
    posy = None
    posz = None
    qua1 = None
    qua2 = None
    qua3 = None
    qua4 = None

  comm.Barrier()
  nmol = comm.bcast(nmol, root=0)
  lx = comm.bcast(lx, root=0)
  ly = comm.bcast(ly, root=0)
  lz = comm.bcast(lz, root=0)
  mid = comm.bcast(mid, root=0)
  posx = comm.bcast(posx, root=0)
  posy = comm.bcast(posy, root=0)
  posz = comm.bcast(posz, root=0)
  qua1 = comm.bcast(qua1, root=0)
  qua2 = comm.bcast(qua2, root=0)
  qua3 = comm.bcast(qua3, root=0)
  qua4 = comm.bcast(qua4, root=0)

  # Make input for MALIO #
  coord = [['' for i in range(3)] for j in range(nmol)]
  direct = [['' for i in range(4)] for j in range(nmol)]
  for i in range(nmol):
    coord[i][0] = posx[i]
    coord[i][1] = posy[i]
    coord[i][2] = posz[i]
    direct[i][0] = qua1[i]
    direct[i][1] = qua2[i]
    direct[i][2] = qua3[i]
    direct[i][3] = qua4[i]
    sim_box = [lx, ly, lz]

  # Use MALIO #
  import pandas as pd

  data_frame = pd.read_csv('__tmp/' + data_frame_prefix + filename_list_p[n] + '.csv', sep=',', header=None, index_col=0)

  import gc
  gc.collect()
  #print("For debug, OP", data_frame)
  op = []
  for i in range(len(data_frame)):
    value = float(data_frame.iat[i,0])
    op.append(value)
    #for i in range(len(op)): # For debug.
    #print(i, op[i]) # List of value of OP for each molecule. Index is an order of reading.

  # Do prediction using .joblib #
  clf_load = joblib.load(joblib_file)
  pred = clf_load.predict(data_frame)
  #print(pred) # List of results of classification for each molecule.

  ### Write classified coordinates. ###
  # Make information for phases. #
  phase = []
  for i in range(len(pred)):
    if pred[i] == 'structure_1':
      phase.append(1)
    if pred[i] == 'structure_2':
      phase.append(2)

  # Write coordinates. #
  if (rank == 0):
    file_data = open(filename_list_w[n], 'w')
    string = [str(nmol), '\n']
    file_data.writelines(string)
    string = [str(lx), ' ', str(ly), ' ', str(lz), '\n']
    file_data.writelines(string)
    for i in range(len(mid)):
      string = [str(mid[i]).ljust(7, ' '), str(phase[i]).ljust(2,' '), ('%12.6f' % coord[i][0]), ('%12.6f' % coord[i][1]), ('%12.6f' % coord[i][2]), ('%12.6f' % direct[i][0]), ('%12.6f' % direct[i][1]), ('%12.6f' % direct[i][2]), ('%12.6f' % direct[i][3]), ('%12.6f' % op[i]), '\n']
      file_data.writelines(string)
    file_data.close()
 
  # Update counter #
  n = n + 1

# Loop end. #

# Write elapsed time. #
t2 = time.time() - t1
if (rank == 0):
  print ("### Record ### Elap.Time:{0}".format(t2) + "[sec]")
