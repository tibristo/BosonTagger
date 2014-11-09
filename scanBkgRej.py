import TaggerTim

paths = []
configs = []
fileids = []


maxrej = 0
maxrejvar = ''
maxrejm_min = 0
maxrejm_max = 0
maxalg = ''
for p,c,f in zip(paths,configs,fileids):
    rej, var, m_min, m_max=TaggerTim.main([c,'-i',p,'-f', f, '--pthigh=2000','--ptlow=350','--nvtx=99','--nvtxlow=0'])
    print 'Algorithm: ' + c
    print 'Rejection: ' + str(rej)
    print 'Variable: ' + var
    print 'Mass min: ' + str(m_min)
    print 'Mass max: ' + str(m_max)
    if rej > maxrej:
        maxrej = rej
        maxrejvar = var
        maxrejm_min = m_min
        maxrejm_max = m_max
        maxalg = c

print '----------------------------------'
print 'Algorithm: ' + maxalg
print 'Rejection: ' + str(maxrej)
print 'Variable: ' + maxrejvar
print 'Mass min: ' + str(maxrejm_min)
print 'Mass max: ' + str(maxrejm_max)
