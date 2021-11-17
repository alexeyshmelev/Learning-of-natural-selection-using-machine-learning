import os
import math
import random
import subprocess
import numpy as np

for i in range(10000):

  #gen_start = 100
  #gen_end = 1000
  #gen = random.randrange(gen_start, gen_end, 100)
  gen = 400
  
  #adm_start = 0.0002
  #adm_end = 0.01
  #l = np.log(adm_end) - np.log(adm_start)
  #adm = round(np.exp(np.random.sample() * l + np.log(adm_start)), 4)
  adm = 0.03

  my_file = open("demography.txt", 'w')
  my_file.write("pop1\tpop2\tsex\t0\t{}\t{}\n0\t0\tF\t10000\t10000\t10000\n0\ta0\tF\t0.5\t0\t0\n0\ta1\tF\t0.5\t{}\t0".format(gen, gen+1, adm))
  my_file.close()

  frc_start = 0.001
  frc_end = 0.1
  l = np.log(frc_end) - np.log(frc_start)
  frc = round(np.exp(np.random.sample() * l + np.log(frc_start)), 4)

  my_file = open("selection.txt", 'w')
  my_file.write("S\tF\t0\t.005\t{}\t{}\t1".format(1-frc, 1-frc/2))
  my_file.close()

  filename = str(gen) + '_' + str(adm) + '_' + str(frc) + '_0.txt'

  if os.path.exists('../next_gen_simulation_usa/' + filename):
    j = 1
    while os.path.exists('../next_gen_simulation_usa/' + filename):
      filename = str(gen) + '_' + str(adm) + '_' + str(frc) + '_' + str(j) + '.txt'
      j += 1

  subprocess.call(["./next_gen_generator.sh", filename])