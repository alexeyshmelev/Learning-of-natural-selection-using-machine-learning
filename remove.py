import os

file_list = os.listdir('absolute path')

for i in range(len(os.listdir('absolute path'))):
    wrong = 0
    file = open('absolute path' + file_list[i], 'r')
    for line in file:
        array = line.split('\t')
        if float(array[2]) == 0:
            wrong = 1
    file.close()
    if wrong == 1:
        os.remove('absolute path' + file_list[i])
