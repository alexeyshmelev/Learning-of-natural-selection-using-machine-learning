import os
import math
import random
import subprocess

for i in range(10):

  gen_start = 1
  gen_end = 500
  log_gen_start = math.log(gen_start)
  log_gen_end = math.log(gen_end)
  gen_l = log_gen_end - log_gen_start
  gen = round(math.exp(random.random() * gen_l + log_gen_start))

  adm_start = 0.0002
  adm_end = 0.01
  log_adm_start = math.log(adm_start)
  log_adm_end = math.log(adm_end)
  adm_l = log_adm_end - log_adm_start
  adm = round(math.exp(random.random() * adm_l + log_adm_start), 4)

  my_file = open("demography.txt", 'w')
  my_file.write("pop1\tpop2\tsex\t0\t{}\t{}\n0\t0\tF\t10000\t10000\t10000\n0\ta0\tF\t0.5\t0\t0\n0\ta1\tF\t0.5\t{}\t0".format(gen, gen+1, adm))
  my_file.close()

  frc_start = 0.0005
  frc_end = 0.02
  log_frc_start = math.log(frc_start)
  log_frc_end = math.log(frc_end)
  frc_l = log_frc_end - log_frc_start
  frc = round(math.exp(random.random() * frc_l + log_frc_start), 4)

  my_file = open("selection.txt", 'w')
  my_file.write("S\tF\t0\t.005\t{}\t{}\t1".format(1-frc, 1-frc/2))
  my_file.close()

  filename = str(gen) + '_0.txt'

  if os.path.exists('next_gen_simulation/' + filename):
    j = 1
    while os.path.exists('next_gen_simulation/' + filename):
      filename = str(gen) + '_' + str(j) + '.txt'
      j += 1

  subprocess.call(["./next_gen_generator.sh", filename])
