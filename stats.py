import os
import matplotlib.pyplot as plt

file_list = os.listdir('absolute path')
adm_list = []
force_list = []

for i in range(len(os.listdir('absolute path'))):
    adm_list += [float(file_list[i].split('_')[1])]
    force_list += [float(file_list[i].split('_')[2])]

plt.subplot(211)
plt.hist(adm_list, label='Admixture')
plt.legend(loc='best')
plt.subplot(212)
plt.hist(force_list, label='Force')
plt.legend(loc='best')
plt.show()
