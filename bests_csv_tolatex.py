import math
import sys
def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs.

    >>> round_sigfigs(0, sig_figs=4)
    0
    >>> int(round_sigfigs(12345, sig_figs=2))
    12000
    >>> int(round_sigfigs(-12345, sig_figs=2))
    -12000
    >>> int(round_sigfigs(1, sig_figs=2))
    1
    >>> '{0:.3}'.format(round_sigfigs(3.1415, sig_figs=2))
    '3.1'
    >>> '{0:.3}'.format(round_sigfigs(-3.1415, sig_figs=2))
    '-3.1'
    >>> '{0:.5}'.format(round_sigfigs(0.00098765, sig_figs=2))
    '0.00099'
    >>> '{0:.6}'.format(round_sigfigs(0.00098765, sig_figs=3))
    '0.000988'
    """
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0




f = open(sys.argv[1],'r')


current = ''
table = '\\begin{table}[h]\n\\small\n\\begin{center}\n\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n'

f_list = list(f)

# write the titles
table += ' & '.join(f_list[0].split(',')).strip().replace('_',' ') + ' \\\\ \n\\hline \n'

# now the top 5 entries
for i in range(1,6):
    table += ' & '.join(f_list[i].split(',')).strip() + ' \\\\ \\hline\n'

table += '\\end{tabular}\n\\caption{}\n\\end{center}\n\\end{table}\n'

f.close()

with open(sys.argv[1].replace('csv','tex').replace('txt','tex'),'w') as t:
    t.write(table)


