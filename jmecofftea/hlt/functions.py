import random
import copy
import coffea.processor as processor
import re
import numpy as np

def match_Jet (ak4, HLT_jet) :
    matched_jet = np.zeros((len(ak4),2))
    for i in range(len(ak4)) :
        dR_lead = 0.2
        dR_sub = 0.2
        lead_index = -1
        sub_index = -1
        print(len(HLT_jet[i]))
        for j in range(len(HLT_jet[i])) :
            dR_lead_tmp = np.sqrt((ak4.eta[i][0]-HLT_jet.eta[i][j])**2 + (ak4.phi[i][0]-HLT_jet.phi[i][j])**2) 
            dR_sub_tmp = np.sqrt((ak4.eta[i][1]-HLT_jet.eta[i][j])**2 + (ak4.phi[i][1]-HLT_jet.phi[i][j])**2) 
            if dR_lead_tmp < dR_lead :
                 dR_lead = dR_lead_tmp
                 lead_index = j
            if dR_sub_tmp < dR_sub :
                 dR_sub = dR_sub_tmp
                 sub_index = j
        matched_jet[i][0]=lead_index
        matched_jet[i][1]=sub_index
        print(str(dR_lead)+" | "+str(dR_sub))
        if i%1000==0 : 
            print(str(i)+"evt")
    return matched_jet

def Tag_and_probe( ak4) :
    tag_ak4 = np.zeros(len(ak4))
    probe_ak4 = np.zeros(len(ak4))
    mask_tnp = []
    for i in range(len(ak4)) :
        if ak4.eta[i][0] < 1.479 & ak4.eta[i][1] < 1.479 :
             mask_tnp.append(True)
             if random.choice([True,False]) :
                 tag_ak4[i] = ak4[i][0]
                 probe_ak4[i] = ak4[i][1]
             else :
                 tag_ak4[i] = ak4[i][1]
                 probe_ak4[i] = ak4[i][0]
        elif ak4.eta[i][0] < 1.479 | ak4.eta[i][1] < 1.479 :
             mask_tnp.append(True)
             if ak4.eta[i][0] < 1.479 :
                 tag_ak4[i] = ak4[i][0]
                 probe_ak4[i] = ak4[i][1]
             else :
                 tag_ak4[i] = ak4[i][1]
                 probe_ak4[i] = ak4[i][0]
        else :
             mask_tnp.append(False)
                 
    return tag_ak4, probe_ak4, mask
