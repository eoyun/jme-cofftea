from klepto.archives import dir_archive
import re
import matplotlib.pyplot as plt
import numpy as np
from coffea import hist
acc = dir_archive("./test/") # Same as the -o argument to jmerge

# Let's say we want to access the MET histogram which we named as "met"
distribution = "ak4_pt0"

acc.load(distribution)
print(acc)

histo = acc[distribution]
print(histo)

for k in histo.values() :
	#print(k[1])

	#h_num = histo.integrate('region','tr_jet_num')
	#h_den = histo.integrate('region','tr_jet_den')
	h_plot = histo.integrate('region',k[1])

	#num = h_num.integrate('dataset',re.compile("Muon.*2023[C]"))
	#den = h_den.integrate('dataset',re.compile("Muon.*2023[C]"))
	plot = h_plot.integrate('dataset',re.compile("Muon.*2023[C]"))

	centers = plot.axes()[0].centers()

	x = np.linspace(min(centers),max(centers),200)

	fig, ax = plt.subplots()
#print("############# print histo ################")
#print(histo.values())
#print("############# print h_numerator ################")
#print(h_num.values())
#print("############# print h_denominator ################")
#print(h_den.values())
#print("############# print numerator ################")
#print(num.values())
#print("############# print denominator ################")
#print(den.values())
	#print(ax)
	hist.plot1d(plot,clear=False)
	fig.savefig('./pngs/'+distribution+"_"+k[1]+'.png')
