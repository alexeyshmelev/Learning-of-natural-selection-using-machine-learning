import os

files = os.listdir('/home/avshmelev/bash_scripts/selam/debugging/sbatches')

for i in files:
    if i != 'new':
        with open('/home/avshmelev/bash_scripts/selam/debugging/sbatches/'+i, 'r') as f:
            with open('/home/avshmelev/bash_scripts/selam/debugging/sbatches/new/'+i.split('.')[0]+'_m.'+i.split('.')[1], 'w') as output:
                for num, line in enumerate(f):
                    if num == 10:
                        output.write('PATH=$PATH:/home/nnshvyrev/SELAM/src\n')
                    elif num == 12:
                        output.write('cd data\n')
                    else:
                        output.write(line)
