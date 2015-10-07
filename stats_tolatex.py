import math
import sys

f = open(sys.argv[1],'r')
f_list = list(f)


split_table = True
title = "DNN"

current = ''
begin_frame = '\\begin{frame}[t]{BDT Cross Validation Sample Statistics} \n'
if title == 'DNN':
    begin_frame = begin_frame.replace('BDT','DNN')
end_frame = '\\end{frame}\n'
begin_table = '\\begin{table}[h]\n\\small\n\\begin{center}\n\\begin{tabular}{|l|c|c|c|c|c|c|}\n'#\\hline\n'
end_table = '\\end{tabular}\n\\caption{Statistics for training variables in the full dataset and the training and validation samples. For the neural network these are standardised such that they have mean of 0 and a standard deviation of 1.}\n\\end{center}\n\\end{table}\n'


ev_table = '\\begin{table}[h]\n\\small\n\\begin{center}\n\\begin{tabular}{|l|c|c|c|}\n\\hline\n'
ev_table += ' & '.join(f_list[0].split()).strip().replace('_',' ') + ' \\\\ \n\\hline \n'

pos = 1
while f_list[pos] != '\n':
    spl = f_list[pos].strip().split() 
    ev_table += (' '.join(spl[:-3])).replace('_',' ') + ' & '
    ev_table += (' & '.join([spl[-3] ,spl[-2], spl[-1]])).replace('_', ' ') + ' \\\\ \n\\hline \n'
    pos+=1
# skip until we get past the '\n'
while f_list[pos] == '\n':
    pos+=1

# this is the varible: mean, std etc. remove the colon!
var_name = f_list[pos].replace(':','')
pos+=2
#table += ' & '.join(var_name.split()).strip().replace('_',' ') + ' \\\\ \n\\hline \n'
table_heading = 'Fold & Sig+Bkg Mean & Sig+Bkg Std & Sig Mean & Sig Std & Bkg Mean & Bkg Std \\\\ \n\\hline \n\\hline \n'
table = begin_frame + begin_table #+ table_heading

# now write all of the different variables!
counter = 0
while pos < len(f_list)-1:
    if f_list[pos] == '\n':
        pos+=1
        pass
    l = f_list[pos].strip().replace(':','')
    spl = l.split()
    if l.find("$") == -1: # if we have an equation here then the underscores are needed, otherwise replace with a space
        variable = l.replace('_','')
    else:
        variable = l
    if len(spl) == 1: # this means that we have found another variable
        if split_table and counter != 0: # write out the first table
            table += end_table + end_frame
            table += begin_frame + begin_table# + table_heading

        table += '\multicolumn{7}{l}{'+variable+ '} \\\\ \\hline\n'
        table += table_heading
        counter += 1
    else:
        name = ' '.join(spl[0:-6]) #+ ' & '
        print name
        
        table += name.replace('_',' ') +' & ' + ' & '.join(spl[len(spl) -6 : ]) + ' \\\\ \\hline\n'
    pos+=1

ev_table += '\\end{tabular}\n\\caption{Number of events in the full dataset and the training and validation samples.}\n\\end{center}\n\\end{table}\n'

#table = begin_frame +  table + end_table + end_frame
table += end_table+end_frame
f.close()

with open(sys.argv[1].replace('csv','tex').replace('txt','tex'),'w') as t:
    t.write(table)

with open(sys.argv[1].replace('.csv','_events.tex').replace('.txt','_events.tex'),'w') as et:
    et.write(ev_table)


