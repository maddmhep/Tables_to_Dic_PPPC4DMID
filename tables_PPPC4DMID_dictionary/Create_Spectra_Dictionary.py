import os,sys
import numpy as np
import math

import matplotlib
from matplotlib  import cm
from matplotlib import ticker
import matplotlib.pyplot as plt



'''
    This function creates a python dictionary called 'Tables' containing all the PPPC spectra without EW corrections,
    contained in the directory AtProductionNoEW_all .
    The X and Masses values have been extracted once for, since they are always fixed
'''
def Create_PPPC_Tables():

    X = [1.1220184543019653e-09, 1.2589254117941663e-09, 1.4125375446227555e-09, 1.584893192461111e-09, 1.7782794100389228e-09, 1.9952623149688828e-09, 2.2387211385683377e-09, 2.511886431509582e-09, 2.8183829312644493e-09, 3.1622776601683795e-09, 3.5481338923357603e-09, 3.981071705534969e-09, 4.466835921509635e-09, 5.011872336272715e-09, 5.623413251903491e-09, 6.309573444801943e-09, 7.0794578438413736e-09, 7.943282347242822e-09, 8.912509381337441e-09, 1e-08, 1.122018454301963e-08, 1.2589254117941661e-08, 1.4125375446227554e-08, 1.5848931924611143e-08, 1.7782794100389228e-08, 1.9952623149688786e-08, 2.2387211385683378e-08, 2.511886431509582e-08, 2.818382931264455e-08, 3.162277660168379e-08, 3.548133892335753e-08, 3.981071705534969e-08, 4.4668359215096346e-08, 5.011872336272725e-08, 5.6234132519034905e-08, 6.30957344480193e-08, 7.079457843841373e-08, 7.943282347242822e-08, 8.912509381337459e-08, 1e-07, 1.122018454301963e-07, 1.2589254117941662e-07, 1.4125375446227555e-07, 1.584893192461114e-07, 1.7782794100389227e-07, 1.9952623149688787e-07, 2.2387211385683377e-07, 2.5118864315095823e-07, 2.818382931264455e-07, 3.162277660168379e-07, 3.548133892335753e-07, 3.981071705534969e-07, 4.466835921509635e-07, 5.011872336272725e-07, 5.62341325190349e-07, 6.30957344480193e-07, 7.079457843841374e-07, 7.943282347242822e-07, 8.912509381337459e-07, 1e-06, 1.122018454301963e-06, 1.2589254117941661e-06, 1.4125375446227554e-06, 1.584893192461114e-06, 1.778279410038923e-06, 1.9952623149688787e-06, 2.2387211385683376e-06, 2.5118864315095823e-06, 2.818382931264455e-06, 3.162277660168379e-06, 3.548133892335753e-06, 3.981071705534969e-06, 4.466835921509635e-06, 5.011872336272725e-06, 5.623413251903491e-06, 6.30957344480193e-06, 7.079457843841373e-06, 7.943282347242822e-06, 8.91250938133746e-06, 1e-05, 1.122018454301963e-05, 1.2589254117941661e-05, 1.4125375446227555e-05, 1.584893192461114e-05, 1.778279410038923e-05, 1.9952623149688786e-05, 2.238721138568338e-05, 2.5118864315095822e-05, 2.818382931264455e-05, 3.1622776601683795e-05, 3.5481338923357534e-05, 3.9810717055349695e-05, 4.466835921509635e-05, 5.011872336272725e-05, 5.623413251903491e-05, 6.309573444801929e-05, 7.079457843841373e-05, 7.943282347242822e-05, 8.912509381337459e-05, 0.0001, 0.0001122018454301963, 0.00012589254117941674, 0.0001412537544622754, 0.00015848931924611142, 0.00017782794100389227, 0.00019952623149688788, 0.000223872113856834, 0.00025118864315095795, 0.0002818382931264455, 0.00031622776601683794, 0.0003548133892335753, 0.00039810717055349735, 0.00044668359215096305, 0.0005011872336272725, 0.0005623413251903491, 0.000630957344480193, 0.000707945784384138, 0.0007943282347242813, 0.0008912509381337459, 0.001, 0.001122018454301963, 0.0012589254117941675, 0.001412537544622754, 0.001584893192461114, 0.0017782794100389228, 0.001995262314968879, 0.00223872113856834, 0.0025118864315095794, 0.002818382931264455, 0.0031622776601683794, 0.0035481338923357532, 0.003981071705534973, 0.0044668359215096305, 0.005011872336272725, 0.005623413251903491, 0.00630957344480193, 0.00707945784384138, 0.007943282347242814, 0.008912509381337459, 0.01, 0.011220184543019636, 0.012589254117941675, 0.01412537544622754, 0.015848931924611134, 0.01778279410038923, 0.0199526231496888, 0.0223872113856834, 0.025118864315095794, 0.028183829312644536, 0.03162277660168379, 0.03548133892335755, 0.039810717055349734, 0.0446683592150963, 0.05011872336272722, 0.05623413251903491, 0.06309573444801933, 0.0707945784384138, 0.07943282347242814, 0.08912509381337455, 0.1, 0.11220184543019636, 0.12589254117941673, 0.14125375446227545, 0.15848931924611134, 0.1778279410038923, 0.19952623149688797, 0.22387211385683395, 0.251188643150958, 0.28183829312644537, 0.31622776601683794, 0.35481338923357547, 0.3981071705534972, 0.44668359215096315, 0.5011872336272722, 0.5623413251903491, 0.6309573444801932, 0.7079457843841379, 0.7943282347242815, 0.8912509381337456, 1.0]
    
    masses = [5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0,   280.0, 300.0, 330.0, 360.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1500.0, 1700.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 12000.0, 15000.0, 20000.0, 30000.0, 50000.0, 100000.0]
    
    spectra_types = ['antiprotons','gammas','neutrinos_e','neutrinos_mu','neutrinos_tau','positrons'] # I skipped the antideuterons
    PyDict =    {'x':X     ,'Masses':masses , 'Particle_Spectra':spectra_types, 'DM_Channels':['ee', 'mumu', 'tautau', 'qq', 'cc', 'bb', 'tt', 'ZZ', 'WW', 'hh', 'gammagamma','gg']}
    PyDict_ew = {'x':X[1:] ,'Masses':masses , 'Particle_Spectra':spectra_types, 'DM_Channels':['ee', 'mumu', 'tautau', 'qq', 'cc', 'bb', 'tt', 'ZZ', 'WW', 'hh', 'gammagamma','gg']}
    # the EW tables have the Log[10,x]=-8.95 missing, so I remove the missing 'x' value from the list of EW Xs
    
    for spec in spectra_types:
        print 'Reading the spectra for the particle : ' , spec
        PyDict[spec]    = {}
        PyDict_ew[spec] = {}
        
        spec_file_noEW = 'AtProductionNoEW_all/AtProductionNoEW_'+spec + '.dat'
        spec_file_EW   = 'AtProduction_all/AtProduction_'+spec + '.dat'

        '''
        # Values extracted with:
        Masses, X = [],[]
        for num in mDM:
            if (num not in Masses):
                Masses.append(num)
        for num in logx:
            N = math.pow(10,num)
            if (N not in X):
                X.append(N)
                    print X
        print X , Masses
        '''

        # Load the data, skipping the header: mDM,Log[10,x],e,\[Mu],\[Tau],q,b,t,W,Z,g,\[Gamma], h
        
        mDM, logx, e, mu, tau, qq, cc, bb, tt, WW, ZZ, gg, gammagamma, hh = np.loadtxt(spec_file_noEW,unpack=True , skiprows=1 )

        # Structure of the dictionary:
        # - 'x' contains the extracted x by calculating N = math.pow(10,num)
        # - 'Masses' is the list of DM masses available
        # - the dictionary spec has the name of the type of spectra (i.e. gamma, positrons etc);
        # - each of them contains a dictionary, name after the DM mass, containing the dN/dlogx values
        
        for mdm in masses:
            #print ' I am reading the mass: \n' , str(mdm)
            dic = { 'ee':[] ,'mumu':[] , 'tautau':[], 'qq':[], 'cc':[] , 'bb' : [] , 'tt':[], 'ZZ':[] , 'WW':[] , 'hh':[] , 'gammagamma':[] ,'gg':[] }
            for L in X:
               for dm, xx, E,M,TA,Q,C,B,T,Z,W,H,gamma,glu in zip(mDM, logx, e, mu, tau, qq, cc, bb, tt, ZZ, WW, hh, gammagamma , gg):
                   xl = math.pow(10,xx)
                   if dm == mdm and L == xl: # double check to make sure to read in the exact order
                      dic['bb']  .append(B)
                      dic['ee']  .append(E)
                      dic['mumu'].append(M)
                      dic['qq']  .append(Q)
                      dic['cc']  .append(C)
                      dic['tt']  .append(T)
                      dic['ZZ']  .append(Z)
                      dic['WW']  .append(W)
                      dic['hh']  .append(H)
                      dic['tautau']      .append(TA)
                      dic['gammagamma']  .append(gamma)
                      dic['gg']  .append(glu)

            PyDict[spec][str(mdm)]=dic
        
        mDM, logx, e, mu, tau, qq, cc, bb, tt, WW, ZZ, gg, gammagamma, hh = np.loadtxt(spec_file_EW,unpack=True , skiprows=1, usecols = (0,1,4,7,10,11,12,13,14,17,20,21,22,23) )

        for mdm in masses:
            #print ' I am reading the mass: \n' , str(mdm)
            dic_ew = { 'ee':[] ,'mumu':[] , 'tautau':[], 'qq':[], 'cc':[],  'bb' : [] , 'tt':[], 'ZZ':[] , 'WW':[] , 'hh':[] , 'gammagamma':[] ,'gg':[] }
            
            for L in X:
               for dm, xx, E,M,TA,Q,B,T,Z,W,H,gamma,glu in zip(mDM, logx, e, mu, tau, qq, bb, tt, ZZ, WW, hh, gammagamma , gg):
                   xl = math.pow(10,xx)
                   if dm == mdm and L == xl: # double check to make sure to read in the exact order
                      dic_ew['bb']  .append(B)
                      dic_ew['ee']  .append(E)
                      dic_ew['mumu'].append(M)
                      dic_ew['qq']  .append(Q)
                      dic_ew['cc']  .append(Q)
                      dic_ew['tt']  .append(T)
                      dic_ew['ZZ']  .append(Z)
                      dic_ew['WW']  .append(W)
                      dic_ew['hh']  .append(H)
                      dic_ew['tautau']      .append(TA)
                      dic_ew['gammagamma']  .append(gamma)
                      dic_ew['gg']  .append(glu)

            PyDict_ew[spec][str(mdm)] = dic_ew

    # Saving the diciotnaries in the npy files
    
    if (os.path.isfile('PPPC_Tables_NoEW.npy')):
            print 'Removing old version of the dictionary and creating a new one'
            os.remove('PPPC_Tables_NoEW.npy')

    np.save('PPPC_Tables_noEW', PyDict)


    if (os.path.isfile('PPPC_Tables_EW.npy')):
            print 'Removing old version of the dictionary and creating a new one'
            os.remove('PPPC_Tables_EW.npy')
        
    np.save('PPPC_Tables_EW', PyDict_ew)



#Create_PPPC_Tables()



    # **********************************************************************
    # How To Use the Dictionary
    # **********************************************************************



Tables = np.load('PPPC_Tables_noEW.npy').item()    # Load the Dictionary from the npy file

# print
print ' ************ PPPPC Dictionary ************ \n'
print 'Available DM candidate Masses:\t\t'        , Tables['Masses']           , '\n'
print 'Available Particle Spectra:\t\t'           , Tables['Particle_Spectra'] , '\n'  #['antiprotons', 'gammas', 'neutrinos_e', 'neutrinos_mu', 'neutrinos_tau', 'positrons']
print 'Available DM annihilation channels:\t'     , Tables['DM_Channels']      , '\n'  #['ee', 'mumu', 'tautau', 'qq', 'cc' 'bb', 'tt', 'ZZ', 'WW', 'hh', 'gammagamma', 'gg']

spectrum , mass , channel = 'gammas', '5.0', 'cc'

print 'The ' + spectrum + ' for DM = ' + mass + ' 5 GeV in the channel DM DM -> ' + channel + ' is: \n'

for x, s in zip(Tables['x'], Tables[spectrum][mass][channel]):
    print x , '\t\t' , s


#print 'Tables = ', Spectra_Reader_NoEW()
