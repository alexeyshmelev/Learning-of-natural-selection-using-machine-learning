import os

files = os.listdir(r'C:\HSE\EPISTASIS\SELAM_copy\patches\sbatches')[1:]

for i in files:
    with open(r'C:\HSE\EPISTASIS\SELAM_copy\patches\sbatches\\'+i, 'r') as f:
        with open(r"C:\HSE\EPISTASIS\SELAM_copy\patches\sbatches\new\\"+i.split('.')[0]+'_m.'+i.split('.')[1], 'w') as output:
            for num, line in enumerate(f):
                if num == 10:
                    output.write("PATH=$PATH:/home/nnshvyrev/SELAM/src\n")
                elif num == 12:
                    output.write("cd data\n")
                else:
                    output.write(line)
