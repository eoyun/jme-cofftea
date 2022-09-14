import matplotlib.pyplot as plt
from coffea import hist
from coffea.util import load
from bucoffea.plot.util import fig_ratio
import numpy as np

from klepto.archives import dir_archive
acc = dir_archive(
                  '/afs/cern.ch/user/j/jbarlow/bucoffea/bucoffea/execute/submission/SingleMuon2022C_2022D/merged_coffea', # Same as the -o argument to bumerge
                  serialized=True,
                  compression=0,
                  memsize=1e3,
                  )

# Get the file
#acc = load('../scripts/hlt_SingleMuon-2017C.coffea')

acc.load('trigger_turnon')
h = acc['trigger_turnon']

#h = acc['met']

# Integrate over the categorical axes of the histogram
# For this example, let's say we have dataset and region axes (typical in our histograms)
#print(h.values())
#h = h.integrate('dataset', 'MET_manyfiles_withtrig-2017C')
#h = h.integrate('region', 'pt0>80, looseid, mftmht_clean_trigger') #integrate out clean trigger
#h = h.integrate('region', 'pt0>80, looseid, mftmht_trigger')

#h1 = h1.integrate('dataset', 'MET-all_tightid_withtrig-2017C')
#h = h.integrate('region', 'pt0>80, looseid, mftmht_clean_trigger') #integrate out clean trigger
#h = h.integrate('region', 'pt0>80, looseid, mftmht_trigger')

#h2 = h2.integrate('dataset', 'dimuon_mass-SingleMuon-2017C')

newax = hist.Bin("turnon", "METnoMu120 (GeV)", np.array(list(range(0,400,20)) + list(range(400,1100,100))))
h = h.rebin(h.axis(newax.name), newax)

#h = h.integrate('dataset', 'trigger-turnon-SingleMuon-2017C')
#num = h.integrate('region', 'turn on numerator')
#den = h.integrate('region', 'turn on denominator')
#clean_num = h.integrate('region', 'clean turn on numerator')

#phi0
#h4 = h4.integrate('dataset', 'SingleMuon-all_tightid_withtrig-2017C')

# Plot the remaining numerical axis
#fig, ax, rax = fig_ratio()

print(h.identifiers('dataset'))
print(h.values())
h = h.integrate('dataset')
num = h.integrate('region', 'turn on numerator')
den = h.integrate('region', 'turn on denominator')
clean_num = h.integrate('region', 'clean turn on numerator')

#fig, ax, rax = fig_ratio()

fig, rax = plt.subplots()

# This will make the ratio plot look better
error_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
    }

error_opts1 = {
        'linestyle':'none',
        'marker': 'x',
        'markersize': 10.,
        'color':'b',
        'elinewidth': 1,
    }

hist.plotratio(
    num,
    den,
    ax=rax,
    unc='clopper-pearson',
    error_opts=error_opts,
    label='METnoMu trigger'
)

hist.plotratio(
    clean_num,
    den,
    ax=rax,
    unc='clopper-pearson',
    error_opts=error_opts1,
    clear=False,
    label='clean METnoMu trigger'
)

#hist.plot1d(h, ax=ax, binwnorm=1)
#hist.plot1d(h, ax=ax)

# Plot the ratio on the bottom pad
#hist.plotratio(
#    num,
#    den,
#    ax=ax, # Plot on the bottom pad
#    unc='clopper-pearson',
#    error_opts=error_opts,
#    label='metmht trigger'
#)

#hist.plotratio(
#    clean_num,
#    den,
#    ax=ax,
#    unc='clopper-pearson',
#    error_opts=error_opts1,
#    label='clean metmht trigger',
#    clear=False
#)

#hist.plot1d(h, ax=ax)

#ax.set_xlabel('Recoil (GeV)', fontsize=12)
#ax.set_xlabel('Leading Jet $\phi$')
#ax.set_ylabel('Events/GeV', fontsize=12)
#ax.legend()
#ax.set_yscale('log')
#ax.set_ylim(10e-3, 10e3)
#ax.set_ylim(0, 1.1)
#ax.set_xlim(0, 700.)
#ax.grid(True)
#ax.axhline(1, ls='--', c='k')

rax.text(0.85, 1.01, '13.6 TeV', 
         fontsize=10, color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes)

rax.text(0., 1.01, r'$W \rightarrow \mu \nu$', 
         fontsize=10, color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes)

rax.set_xlabel('Recoil (GeV)')
rax.set_ylabel('Efficiency')
rax.set_ylim(0, 1.1)
rax.set_xlim(0, 700.)
rax.grid(True)
rax.axhline(1, ls='--', c='k')
rax.legend()


fig.savefig('Wmunu-trigeff-SingleMuon-2022D.pdf')
