def ppeDir(fname):
    f = open(fname,'r')
    fname2 = fname[:-4]+'_ppe.lst'
    f2 = open(fname2,'w')

    for l in f:
        idx = len(l) - 1 - l[::-1].index('/')
        f2.write(l[idx:])
    f.close()
    f2.close()

files = ['listMCFiles.Dibosons_AFII.lst','listMCFiles.DibosonsHerwig_AFII.lst','listMCFiles.STop_AFII.lst','listMCFiles.Top_AFII.lst','listMCFiles.WJetsenu_AFII.lst','listMCFiles.WJetsenu_FS.lst','listMCFiles.WJetsmunu_AFII.lst','listMCFiles.WJetsmunu_FS.lst','listMCFiles.WJetstaunu_AFII.lst','listMCFiles.WJetstaunu_FS.lst']

for f in files:
    ppeDir(f)
