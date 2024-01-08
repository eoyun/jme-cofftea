from klepto.archives import dir_archive
import re
import matplotlib.pyplot as plt
import numpy as np
from coffea import hist
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('distribution', type=str, help='The processor to be run. (monojet or vbfhinv)')
parser.add_argument('output', type=str, help='output directory')
args = parser.parse_args()


directory = "output/"+args.output
acc = dir_archive("./"+directory+"") # Same as the -o argument to jmerge

# Let's say we want to access the MET histogram which we named as "met"
distribution = args.distribution

acc.load(distribution)
print(acc)

histo = acc[distribution]
print(histo)
regions = histo.identifiers("region")
pngdir = "./pngs/"+directory
if not os.path.exists(pngdir) :
	os.mkdir(pngdir)

for region in regions :
	print(region)

	#h_num = histo.integrate('region','tr_jet_num')
	#h_den = histo.integrate('region','tr_jet_den')
	h_plot = histo.integrate('region',region.name)

	#num = h_num.integrate('dataset',re.compile("Muon.*2023[C]"))
	#den = h_den.integrate('dataset',re.compile("Muon.*2023[C]"))
	plot = h_plot.integrate('dataset',re.compile("Muon.*2023*"))

	centers = plot.axes()[0].centers()
	if region.name == 'tr_fail_ht1050' :
		continue
	x = np.linspace(min(centers),max(centers),200)
#	print(x)
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
	hist.plot1d(plot)
	fig.savefig('./pngs/'+directory+'/'+distribution+"_"+region.name+'.png')
