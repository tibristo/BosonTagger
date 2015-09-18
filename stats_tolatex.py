import math
import sys

f = open(sys.argv[1],'r')


current = ''
table = '\\begin{table}[h]\n\\small\n\\begin{center}\n\\begin{tabular}{|l|c|c|c|c|c|}\n\\hline\n'

f_list = list(f)

ev_table = '\\begin{table}[h]\n\\small\n\\begin{center}\n\\begin{tabular}{|l|c|c|c|}\n\\hline\n'
ev_table += ' & '.join(f_list[0].split()).strip().replace('_',' ') + ' \\\\ \n\\hline \n'

pos = 1
while f_list[pos] != '\n':
    spl = f_list[pos].strip().split() 
    ev_table += ' '.join(spl[:-3]) + ' & '
    ev_table += ' & '.join([spl[-3] ,spl[-2], spl[-1]]) + ' \\\\ \n\\hline \n'
    pos+=1
# skip until we get past the '\n'
while f_list[pos] == '\n':
    pos+=1

# this is the varible: mean, std etc. remove the colon!
var_name = f_list[pos].replace(':','')
pos+=2
table += ' & '.join(var_name.split()).strip().replace('_',' ') + ' \\\\ \n\\hline \n'

# now write all of the different variables!
while pos < len(f_list)-1:
    if f_list[pos] == '\n':
        pos+=1
        pass
    l = f_list[pos].strip().replace(':','')
    spl = l.split()
    if len(spl) == 1:
        table += 'multicolumn{6}{c}{'+l.replace('_',' ')+ '} \\\\ \\hline\n'
    else:
        name = ' '.join(spl[0:-5]) + ' & '
        table += name + ' & '.join(spl[len(spl) -5 : ]) + ' \\\\ \\hline\n'
    pos+=1

ev_table += '\\end{tabular}\n\\caption{}\n\\end{center}\n\\end{table}\n'
table += '\\end{tabular}\n\\caption{}\n\\end{center}\n\\end{table}\n'

f.close()

with open(sys.argv[1].replace('csv','tex').replace('txt','tex'),'w') as t:
    t.write(table)

with open(sys.argv[1].replace('.csv','_events.tex').replace('.txt','_events.tex'),'w') as et:
    et.write(ev_table)


