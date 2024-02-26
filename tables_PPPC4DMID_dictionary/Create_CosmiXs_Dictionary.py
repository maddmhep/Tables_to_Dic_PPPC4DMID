import os
import os.path
import numpy as np
import pandas as pd

spectra_types = ["AntiP", "Gamma", "Nuel", "Numu", "Nuta", "Positrons"]
map_spectra_types = {
    "AntiP": "antiprotons",
    "Gamma": "gammas",
    "Nuel": "neutrinos_e",
    "Numu": "neutrinos_mu",
    "Nuta": "neutrinos_tau",
    "Positrons": "positrons",
}
channels = [
    "eL",
    "eR",
    "e",
    "muL",
    "muR",
    "mu",
    "tauL",
    "tauR",
    "tau",
    "q",
    "c",
    "b",
    "t",
    "a",
    "g",
    "W",
    "Z",
    "H",
    "aZ",
    "HZ",
    "nue",
    "numu",
    "nutau",
]
light_quarks = ["u", "d", "s"]
map_channels = {
    "eL": "eL",
    "eR": "eR",
    "e": "ee",
    "muL": "muL",
    "muR": "muR",
    "mu": "mumu",
    "tauL": "tauL",
    "tauR": "tauR",
    "tau": "tautau",
    "q": "qq",
    "c": "cc",
    "b": "bb",
    "t": "tt",
    "a": "gammagamma",
    "g": "gg",
    "W": "WW",
    "Z": "ZZ",
    "H": "hh",
    "aZ": "gammaz",
    "HZ": "hZ",
    "nue": "nu_e",
    "numu": "nu_mu",
    "nutau": "nu_tau",
}

names = [
    "mass",
    "log10x",
    "eL",
    "eR",
    "e",
    "muL",
    "muR",
    "mu",
    "tauL",
    "tauR",
    "tau",
    "nue",
    "numu",
    "nutau",
    "u",
    "d",
    "s",
    "c",
    "b",
    "t",
    "a",
    "g",
    "W",
    "WL",
    "WT",
    "Z",
    "ZL",
    "ZT",
    "H",
    "aZ",
    "HZ"
]
PyDict = {
    "Particle_Spectra": list(map_spectra_types.values()),
    "DM_Channels": list(map_channels.values()),
    "Masses": [],
}

for i, spec in enumerate(spectra_types):
    spec_file = os.path.join("CosmiXs", "Data", "AtProduction-" + spec + ".dat")
    df = pd.read_csv(spec_file, names=names, header=1, sep=r"\s+")
    df["q"] = np.sum(
        np.array([df[light_quark].values for light_quark in light_quarks]), axis=0
    )

    PyDict[map_spectra_types[spec]] = {}
    for j, (mdm, group_df) in enumerate(df.groupby("mass")):
        if j == 1:
            PyDict["x"] = np.power(10, group_df["log10x"].values).tolist()
        if i == 1:
            PyDict["Masses"].append(mdm)
        PyDict[map_spectra_types[spec]][str(mdm)] = {
            v: group_df[k].values.tolist() for k, v in map_channels.items()
        }

# Saving the dictionaries in the npy files
FILENAME = os.path.join("PPPC4DMID", "CosmiXs.npy")
if os.path.isfile(FILENAME):
    os.remove(FILENAME)

np.save(FILENAME, PyDict)
