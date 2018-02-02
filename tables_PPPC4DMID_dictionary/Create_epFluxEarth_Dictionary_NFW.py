import os,sys
import numpy as np
import math

import matplotlib
from matplotlib  import cm
from matplotlib import ticker
import matplotlib.pyplot as plt



'''
    This function creates a python dictionary called 'Tables' containing all the PPPC spectra for positrons at detection after propagation.
'''
def Spectra_Reader_epEarth(create = True):

    # kinetic energy of the positron (stored as Log10(E))
    X = [-1.0, -0.90000000000000002, -0.80000000000000004, -0.69999999999999996, -0.59999999999999998, -0.5, -0.40000000000000002, -0.29999999999999999, -0.20000000000000001,
             -0.10000000000000001, 0.0, 0.10000000000000001, 0.20000000000000001, 0.29999999999999999, 0.40000000000000002, 0.5, 0.59999999999999998, 0.69999999999999996,
             0.80000000000000004, 0.90000000000000002, 1.0, 1.1000000000000001, 1.2, 1.3, 1.3999999999999999, 1.5, 1.6000000000000001, 1.7, 1.8, 1.8999999999999999, 2.0,
             2.1000000000000001, 2.2000000000000002, 2.2999999999999998, 2.3999999999999999, 2.5, 2.6000000000000001, 2.7000000000000002, 2.7999999999999998, 2.8999999999999999, 3.0,
             3.1000000000000001, 3.2000000000000002, 3.2999999999999998, 3.3999999999999999, 3.5, 3.6000000000000001, 3.7000000000000002, 3.7999999999999998, 3.8999999999999999, 4.0,
             4.0999999999999996, 4.2000000000000002, 4.2999999999999998, 4.4000000000000004, 4.5, 4.5999999999999996, 4.7000000000000002, 4.7999999999999998, 4.9000000000000004, 5.0]
    
    #  DM mass
    masses = [5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0,   280.0, 300.0, 330.0, 360.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1500.0, 1700.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 12000.0, 15000.0, 20000.0, 30000.0, 50000.0, 100000.0]

    #
    halo = ['NFW'] # 'Iso', 'Ein',  'Moo' !! 'Bur','EiB'

    prop = ['MIN', 'MED', 'MAX']

    MF = ['MF1', 'MF2', 'MF3']

    channels = ['ee_NFW_MIN_MF1','ee_NFW_MIN_MF2','ee_NFW_MIN_MF3','ee_NFW_MED_MF1','ee_NFW_MED_MF2','ee_NFW_MED_MF3','ee_NFW_MAX_MF1','ee_NFW_MAX_MF2','ee_NFW_MAX_MF3',
                    'mumu_NFW_MIN_MF1','mumu_NFW_MIN_MF2','mumu_NFW_MIN_MF3','mumu_NFW_MED_MF1','mumu_NFW_MED_MF2','mumu_NFW_MED_MF3','mumu_NFW_MAX_MF1','mumu_NFW_MAX_MF2','mumu_NFW_MAX_MF3',
                    'tautau_NFW_MIN_MF1','tautau_NFW_MIN_MF2','tautau_NFW_MIN_MF3','tautau_NFW_MED_MF1','tautau_NFW_MED_MF2','tautau_NFW_MED_MF3','tautau_NFW_MAX_MF1','tautau_NFW_MAX_MF2','tautau_NFW_MAX_MF3',
                    'qq_NFW_MIN_MF1','qq_NFW_MIN_MF2','qq_NFW_MIN_MF3','qq_NFW_MED_MF1','qq_NFW_MED_MF2','qq_NFW_MED_MF3','qq_NFW_MAX_MF1','qq_NFW_MAX_MF2','qq_NFW_MAX_MF3',
                    'cc_NFW_MIN_MF1','cc_NFW_MIN_MF2','cc_NFW_MIN_MF3','cc_NFW_MED_MF1','cc_NFW_MED_MF2','cc_NFW_MED_MF3','cc_NFW_MAX_MF1','cc_NFW_MAX_MF2','cc_NFW_MAX_MF3',
                    'bb_NFW_MIN_MF1','bb_NFW_MIN_MF2','bb_NFW_MIN_MF3','bb_NFW_MED_MF1','bb_NFW_MED_MF2','bb_NFW_MED_MF3','bb_NFW_MAX_MF1','bb_NFW_MAX_MF2','bb_NFW_MAX_MF3',
                    'tt_NFW_MIN_MF1','tt_NFW_MIN_MF2','tt_NFW_MIN_MF3','tt_NFW_MED_MF1','tt_NFW_MED_MF2','tt_NFW_MED_MF3','tt_NFW_MAX_MF1','tt_NFW_MAX_MF2','tt_NFW_MAX_MF3',
                    'ZZ_NFW_MIN_MF1','ZZ_NFW_MIN_MF2','ZZ_NFW_MIN_MF3','ZZ_NFW_MED_MF1','ZZ_NFW_MED_MF2','ZZ_NFW_MED_MF3','ZZ_NFW_MAX_MF1','ZZ_NFW_MAX_MF2','ZZ_NFW_MAX_MF3',
                    'WW_NFW_MIN_MF1','WW_NFW_MIN_MF2','WW_NFW_MIN_MF3','WW_NFW_MED_MF1','WW_NFW_MED_MF2','WW_NFW_MED_MF3','WW_NFW_MAX_MF1','WW_NFW_MAX_MF2','WW_NFW_MAX_MF3',
                    'hh_NFW_MIN_MF1','hh_NFW_MIN_MF2','hh_NFW_MIN_MF3','hh_NFW_MED_MF1','hh_NFW_MED_MF2','hh_NFW_MED_MF3','hh_NFW_MAX_MF1','hh_NFW_MAX_MF2','hh_NFW_MAX_MF3',
                    'gg_NFW_MIN_MF1','gg_NFW_MIN_MF2','gg_NFW_MIN_MF3','gg_NFW_MED_MF1','gg_NFW_MED_MF2','gg_NFW_MED_MF3','gg_NFW_MAX_MF1','gg_NFW_MAX_MF2','gg_NFW_MAX_MF3',
                    'aa_NFW_MIN_MF1','aa_NFW_MIN_MF2','aa_NFW_MIN_MF3','aa_NFW_MED_MF1','aa_NFW_MED_MF2','aa_NFW_MED_MF3','aa_NFW_MAX_MF1','aa_NFW_MAX_MF2','aa_NFW_MAX_MF3']
    
    spectra_types = ['positrons'] 
    PyDict = {'x':X ,'Masses':masses, 'DM_Channels':channels,'Prop_particles': spectra_types} 
    
    for spec in spectra_types:
        print 'Defining the dictionary for the flux at Earth of: ' , spec
        PyDict[spec] = {}
        spec_file = 'AfterPropagation_Ann_positrons/AfterPropagation_Ann_' + spec + '.dat'
        
        # Load the data, skipping the header: mDM,Log[10,x],e,\[Mu],\[Tau],q,b,t,W,Z,g,\[Gamma], h
        mDM, halodf, propdf, mfdf, logx, e, mu, tau, qq, cc, bb, tt, WW, ZZ, gg, aa, hh = np.loadtxt(spec_file,unpack=True,
                                                                                                         skiprows=1,usecols=(0,1,2,3,4,7,10,13,14,15,16,17,20,23,24,25,26),
                                                                                                         dtype=np.dtype([('mDM','<f8'),('halo','S3'),('prop','S3'),('mfi','S3'),('logx','<f8'),
                                                                                                                             ('e','<f8'),('mu','<f8'),('tau','<f8'),('qq','<f8'),
                                                                                                                             ('cc','<f8'),('bb','<f8'),('tt','<f8'),('WW','<f8'),('ZZ','<f8'),
                                                                                                                             ('gg','<f8'),('aa','<f8'),('hh','<f8')]))
                
        # Structure of the dictionary:
        # - 'x' contains the extracted x 
        # - 'Masses' is the list of DM masses available
        # - the dictionary spec has the name of the propagated particle (i.e. positrons or anti-protons);
        #   each of them contains a dictionary, named after the DM mass
        #   containing the dPhi/dE values relative the DM density profile, propagation model and mass function model.
        
        for mdm in masses:
            print 'creating tables for DM mass = ', str(mdm)
            dic = {'ee_NFW_MIN_MF1':[],'mumu_NFW_MIN_MF1':[] , 'tautau_NFW_MIN_MF1':[], 'qq_NFW_MIN_MF1':[], 'cc_NFW_MIN_MF1':[], 'bb_NFW_MIN_MF1':[] ,
                       'tt_NFW_MIN_MF1':[], 'ZZ_NFW_MIN_MF1':[] , 'WW_NFW_MIN_MF1':[] , 'hh_NFW_MIN_MF1':[], 'aa_NFW_MIN_MF1':[],'gg_NFW_MIN_MF1':[],
                       'ee_NFW_MED_MF1':[],'mumu_NFW_MED_MF1':[] , 'tautau_NFW_MED_MF1':[], 'qq_NFW_MED_MF1':[], 'cc_NFW_MED_MF1':[], 'bb_NFW_MED_MF1':[],
                       'tt_NFW_MED_MF1':[], 'ZZ_NFW_MED_MF1':[] , 'WW_NFW_MED_MF1':[] , 'hh_NFW_MED_MF1':[], 'aa_NFW_MED_MF1':[],'gg_NFW_MED_MF1':[],
                       'ee_NFW_MAX_MF1':[],'mumu_NFW_MAX_MF1':[] , 'tautau_NFW_MAX_MF1':[], 'qq_NFW_MAX_MF1':[], 'cc_NFW_MAX_MF1':[], 'bb_NFW_MAX_MF1':[],
                       'tt_NFW_MAX_MF1':[], 'ZZ_NFW_MAX_MF1':[] , 'WW_NFW_MAX_MF1':[] , 'hh_NFW_MAX_MF1':[], 'aa_NFW_MAX_MF1':[],'gg_NFW_MAX_MF1':[],
                    'ee_NFW_MIN_MF2':[],'mumu_NFW_MIN_MF2':[] , 'tautau_NFW_MIN_MF2':[], 'qq_NFW_MIN_MF2':[], 'cc_NFW_MIN_MF2':[], 'bb_NFW_MIN_MF2':[],
                       'tt_NFW_MIN_MF2':[], 'ZZ_NFW_MIN_MF2':[] , 'WW_NFW_MIN_MF2':[] , 'hh_NFW_MIN_MF2':[], 'aa_NFW_MIN_MF2':[],'gg_NFW_MIN_MF2':[],
                       'ee_NFW_MED_MF2':[],'mumu_NFW_MED_MF2':[] , 'tautau_NFW_MED_MF2':[], 'qq_NFW_MED_MF2':[], 'cc_NFW_MED_MF2':[], 'bb_NFW_MED_MF2':[], 
                       'tt_NFW_MED_MF2':[], 'ZZ_NFW_MED_MF2':[] , 'WW_NFW_MED_MF2':[] , 'hh_NFW_MED_MF2':[], 'aa_NFW_MED_MF2':[],'gg_NFW_MED_MF2':[],
                       'ee_NFW_MAX_MF2':[],'mumu_NFW_MAX_MF2':[] , 'tautau_NFW_MAX_MF2':[], 'qq_NFW_MAX_MF2':[], 'cc_NFW_MAX_MF2':[], 'bb_NFW_MAX_MF2':[],
                       'tt_NFW_MAX_MF2':[], 'ZZ_NFW_MAX_MF2':[] , 'WW_NFW_MAX_MF2':[] , 'hh_NFW_MAX_MF2':[], 'aa_NFW_MAX_MF2':[],'gg_NFW_MAX_MF2':[],
                     'ee_NFW_MIN_MF3':[],'mumu_NFW_MIN_MF3':[] , 'tautau_NFW_MIN_MF3':[], 'qq_NFW_MIN_MF3':[], 'cc_NFW_MIN_MF3':[], 'bb_NFW_MIN_MF3':[],
                       'tt_NFW_MIN_MF3':[], 'ZZ_NFW_MIN_MF3':[] , 'WW_NFW_MIN_MF3':[] , 'hh_NFW_MIN_MF3':[], 'aa_NFW_MIN_MF3':[],'gg_NFW_MIN_MF3':[],
                       'ee_NFW_MED_MF3':[],'mumu_NFW_MED_MF3':[] , 'tautau_NFW_MED_MF3':[], 'qq_NFW_MED_MF3':[], 'cc_NFW_MED_MF3':[], 'bb_NFW_MED_MF3':[],
                       'tt_NFW_MED_MF3':[], 'ZZ_NFW_MED_MF3':[] , 'WW_NFW_MED_MF3':[] , 'hh_NFW_MED_MF3':[], 'aa_NFW_MED_MF3':[],'gg_NFW_MED_MF3':[],
                       'ee_NFW_MAX_MF3':[],'mumu_NFW_MAX_MF3':[] , 'tautau_NFW_MAX_MF3':[], 'qq_NFW_MAX_MF3':[], 'cc_NFW_MAX_MF3':[], 'bb_NFW_MAX_MF3':[],
                       'tt_NFW_MAX_MF3':[], 'ZZ_NFW_MAX_MF3':[] , 'WW_NFW_MAX_MF3':[] , 'hh_NFW_MAX_MF3':[], 'aa_NFW_MAX_MF3':[],'gg_NFW_MAX_MF3':[],
                       }

                
            for halodm in halo:
                for propmod in prop:
                    for mfimod in MF:
                        for L in X:
                            for dm,hdm,propcr,mfun,xx,E,M,TA,Q,C,B,T,Z,W,H,gamma,glu in zip(mDM, halodf, propdf, mfdf, logx, e, mu, tau, qq, cc, bb, tt, ZZ, WW, hh, aa, gg): 
                                if dm == mdm and L == xx:
                                    if hdm == halodm and halodm == 'NFW':
                                        if propcr == propmod and propmod == 'MIN':
                                            if mfun == mfimod and mfimod == 'MF1':
                                                dic['bb_NFW_MIN_MF1'].append(B)
                                                dic['ee_NFW_MIN_MF1'].append(E)
                                                dic['mumu_NFW_MIN_MF1'].append(M)
                                                dic['qq_NFW_MIN_MF1'].append(Q)
                                                dic['cc_NFW_MIN_MF1'].append(C)
                                                dic['tt_NFW_MIN_MF1'].append(T)
                                                dic['ZZ_NFW_MIN_MF1'].append(Z)
                                                dic['WW_NFW_MIN_MF1'].append(W)
                                                dic['hh_NFW_MIN_MF1'].append(H)
                                                dic['tautau_NFW_MIN_MF1'].append(TA)
                                                dic['aa_NFW_MIN_MF1'].append(gamma)
                                                dic['gg_NFW_MIN_MF1'].append(glu)
                                            elif mfun == mfimod and mfimod == 'MF2':
                                                dic['bb_NFW_MIN_MF2'].append(B)
                                                dic['ee_NFW_MIN_MF2'].append(E)
                                                dic['mumu_NFW_MIN_MF2'].append(M)
                                                dic['qq_NFW_MIN_MF2'].append(Q)
                                                dic['cc_NFW_MIN_MF2'].append(C)
                                                dic['tt_NFW_MIN_MF2'].append(T)
                                                dic['ZZ_NFW_MIN_MF2'].append(Z)
                                                dic['WW_NFW_MIN_MF2'].append(W)
                                                dic['hh_NFW_MIN_MF2'].append(H)
                                                dic['tautau_NFW_MIN_MF2'].append(TA)
                                                dic['aa_NFW_MIN_MF2'].append(gamma)
                                                dic['gg_NFW_MIN_MF2'].append(glu)
                                            elif mfun == mfimod and mfimod == 'MF3':
                                                dic['bb_NFW_MIN_MF3'].append(B)
                                                dic['ee_NFW_MIN_MF3'].append(E)
                                                dic['mumu_NFW_MIN_MF3'].append(M)
                                                dic['qq_NFW_MIN_MF3'].append(Q)
                                                dic['cc_NFW_MIN_MF3'].append(C)
                                                dic['tt_NFW_MIN_MF3'].append(T)
                                                dic['ZZ_NFW_MIN_MF3'].append(Z)
                                                dic['WW_NFW_MIN_MF3'].append(W)
                                                dic['hh_NFW_MIN_MF3'].append(H)
                                                dic['tautau_NFW_MIN_MF3'].append(TA)
                                                dic['aa_NFW_MIN_MF3'].append(gamma)
                                                dic['gg_NFW_MIN_MF3'].append(glu)
                                        elif propcr == propmod and propmod == 'MED':
                                            if mfun == mfimod and mfimod == 'MF1':
                                                dic['bb_NFW_MED_MF1'].append(B)
                                                dic['ee_NFW_MED_MF1'].append(E)
                                                dic['mumu_NFW_MED_MF1'].append(M)
                                                dic['qq_NFW_MED_MF1'].append(Q)
                                                dic['cc_NFW_MED_MF1'].append(C)
                                                dic['tt_NFW_MED_MF1'].append(T)
                                                dic['ZZ_NFW_MED_MF1'].append(Z)
                                                dic['WW_NFW_MED_MF1'].append(W)
                                                dic['hh_NFW_MED_MF1'].append(H)
                                                dic['tautau_NFW_MED_MF1'].append(TA)
                                                dic['aa_NFW_MED_MF1'].append(gamma)
                                                dic['gg_NFW_MED_MF1'].append(glu)
                                            elif mfun == mfimod and mfimod == 'MF2':
                                                dic['bb_NFW_MED_MF2'].append(B)
                                                dic['ee_NFW_MED_MF2'].append(E)
                                                dic['mumu_NFW_MED_MF2'].append(M)
                                                dic['qq_NFW_MED_MF2'].append(Q)
                                                dic['cc_NFW_MED_MF2'].append(C)
                                                dic['tt_NFW_MED_MF2'].append(T)
                                                dic['ZZ_NFW_MED_MF2'].append(Z)
                                                dic['WW_NFW_MED_MF2'].append(W)
                                                dic['hh_NFW_MED_MF2'].append(H)
                                                dic['tautau_NFW_MED_MF2'].append(TA)
                                                dic['aa_NFW_MED_MF2'].append(gamma)
                                                dic['gg_NFW_MED_MF2'].append(glu)
                                            elif mfun == mfimod and mfimod == 'MF3':
                                                dic['bb_NFW_MED_MF3'].append(B)
                                                dic['ee_NFW_MED_MF3'].append(E)
                                                dic['mumu_NFW_MED_MF3'].append(M)
                                                dic['qq_NFW_MED_MF3'].append(Q)
                                                dic['cc_NFW_MED_MF3'].append(C)
                                                dic['tt_NFW_MED_MF3'].append(T)
                                                dic['ZZ_NFW_MED_MF3'].append(Z)
                                                dic['WW_NFW_MED_MF3'].append(W)
                                                dic['hh_NFW_MED_MF3'].append(H)
                                                dic['tautau_NFW_MED_MF3'].append(TA)
                                                dic['aa_NFW_MED_MF3'].append(gamma)
                                                dic['gg_NFW_MED_MF3'].append(glu)
                                        elif propcr == propmod and propmod == 'MAX':
                                            if mfun == mfimod and mfimod == 'MF1':
                                               dic['bb_NFW_MAX_MF1'].append(B)
                                               dic['ee_NFW_MAX_MF1'].append(E)
                                               dic['mumu_NFW_MAX_MF1'].append(M)
                                               dic['qq_NFW_MAX_MF1'].append(Q)
                                               dic['cc_NFW_MAX_MF1'].append(C)
                                               dic['tt_NFW_MAX_MF1'].append(T)
                                               dic['ZZ_NFW_MAX_MF1'].append(Z)
                                               dic['WW_NFW_MAX_MF1'].append(W)
                                               dic['hh_NFW_MAX_MF1'].append(H)
                                               dic['tautau_NFW_MAX_MF1'].append(TA)
                                               dic['aa_NFW_MAX_MF1'].append(gamma)
                                               dic['gg_NFW_MAX_MF1'].append(glu)
                                            elif mfun == mfimod and mfimod == 'MF2':
                                                dic['bb_NFW_MAX_MF2'].append(B)
                                                dic['ee_NFW_MAX_MF2'].append(E)
                                                dic['mumu_NFW_MAX_MF2'].append(M)
                                                dic['qq_NFW_MAX_MF2'].append(Q)
                                                dic['cc_NFW_MAX_MF2'].append(C)
                                                dic['tt_NFW_MAX_MF2'].append(T)
                                                dic['ZZ_NFW_MAX_MF2'].append(Z)
                                                dic['WW_NFW_MAX_MF2'].append(W)
                                                dic['hh_NFW_MAX_MF2'].append(H)
                                                dic['tautau_NFW_MAX_MF2'].append(TA)
                                                dic['aa_NFW_MAX_MF2'].append(gamma)
                                                dic['gg_NFW_MAX_MF2'].append(glu)
                                            elif mfun == mfimod and mfimod == 'MF3':
                                                dic['bb_NFW_MAX_MF3'].append(B)
                                                dic['ee_NFW_MAX_MF3'].append(E)
                                                dic['mumu_NFW_MAX_MF3'].append(M)
                                                dic['qq_NFW_MAX_MF3'].append(Q)
                                                dic['cc_NFW_MAX_MF3'].append(C)
                                                dic['tt_NFW_MAX_MF3'].append(T)
                                                dic['ZZ_NFW_MAX_MF3'].append(Z)
                                                dic['WW_NFW_MAX_MF3'].append(W)
                                                dic['hh_NFW_MAX_MF3'].append(H)
                                                dic['tautau_NFW_MAX_MF3'].append(TA)
                                                dic['aa_NFW_MAX_MF3'].append(gamma)
                                                dic['gg_NFW_MAX_MF3'].append(glu)
                                    
                                                
            PyDict[spec][str(mdm)]=dic

    if create:
        if (os.path.isfile('PPPC_Tables_epEarth_NFW.npy')):
            print 'Removing old version of the dictionary and creating a new one'
            os.remove('PPPC_Tables_epEarth_NFW.npy')
        np.save('PPPC_Tables_epEarth_NFW', PyDict)

# Create the dictionary

# Spectra_Reader_epEarth(create = True)              # This creates the numpy file containing the dictionary

    # **********************************************************************
    # How To Use the Dictionary
    # **********************************************************************
    
#Tables = np.load('PPPC_Tables_epEarth.npy').item()    # Load the Dictionary from the file

#print
#print ' ************ PPPPC Dictionary e+ at Earth ************ \n'
#print 'Available DM candidate Masses:\t\t'        , Tables['Masses']           , '\n'
#print 'Available Propagated particles:\t\t'           , Tables['Prop_particles'] , '\n'
#print 'Available DM annihilation channels:\t'     , Tables['DM_Channels']      , '\n'

#ep, mass, channel = 'positrons', '6.0', 'ee_Moo_MAX_MF3'#, 'NFW', 'MIN', 'MF1'  , halo_model, prop_model, mf_model

#print 'The ' + ep + ' flux at Earth for DM = ' + mass + 'GeV in the channel DM DM -> ' + channel + ' is: \n'

#for x, s in zip(Tables['x'], Tables[ep][mass][channel]): #[halo_model][prop_model][mf_model]
#    print x , '\t\t' , s


print 'Tables = ', Spectra_Reader_epEarth()

