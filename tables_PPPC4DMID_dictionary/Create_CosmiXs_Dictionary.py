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
# Add AntiD label
map_spectra_types["AntiD"] = "antideuterons"

# Only keep desired channels
map_channels = {
    "e": "ee",
    "mu": "mumu",
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
    "nue": "nu_e",
    "numu": "nu_mu",
    "nutau": "nu_tau",
}

# Light quarks for q = u + d + s
light_quarks = ["u", "d", "s"]

# Full column names (required for reading files)
names = [
    "mass", "log10x",
    "eL", "eR", "e", "muL", "muR", "mu", "tauL", "tauR", "tau",
    "nue", "numu", "nutau",
    "u", "d", "s", "c", "b", "t",
    "a", "g",
    "W", "WL", "WT", "Z", "ZL", "ZT",
    "H", "aZ", "HZ"
]

PyDict = {
    "Particle_Spectra": list(map_spectra_types.values()),
    "DM_Channels": list(map_channels.values()),
    "Masses": [],
}

# --- Loop over original spectra types ---
for i, spec in enumerate(spectra_types):
    spec_file = os.path.join("CosmiXs", "Data", f"AtProduction-{spec}.dat")
    df = pd.read_csv(spec_file, names=names, header=1, sep=r"\s+")

    # Add "q" as sum of light quarks
    df["q"] = np.sum([df[qk].values for qk in light_quarks], axis=0)

    PyDict[map_spectra_types[spec]] = {}
    for j, (mdm, group_df) in enumerate(df.groupby("mass")):
        if j == 1:
            PyDict["x"] = np.power(10, group_df["log10x"].values).tolist()
        if i == 1:
            PyDict["Masses"].append(mdm)

        # Only keep selected channels
        PyDict[map_spectra_types[spec]][str(mdm)] = {
            v: group_df[k].values.tolist() for k, v in map_channels.items() if k in group_df.columns
        }

# --- Now read AntiD separately ---
antid_file = os.path.join("CosmiXs", "Data", "AtProduction-AntiD-AWF.dat")

# Custom column names for AntiD file (derived from your list)
antid_names = [
    "mass", "log10x", "u", "d", "s", "c", "b", "t",
    "a", "g", "W", "WL", "WT", "Z", "ZL", "ZT", "H", "aZ", "HZ"
]
df_antid = pd.read_csv(antid_file, names=antid_names, header=1, sep=r"\s+")

# Add q = u + d + s
df_antid["q"] = df_antid["u"] + df_antid["d"] + df_antid["s"]

PyDict["antideuterons"] = {}
for j, (mdm, group_df) in enumerate(df_antid.groupby("mass")):
    spectra_dict = {}
    for k, v in map_channels.items():
        if k in group_df.columns:
            spectra_dict[v] = group_df[k].values.tolist()
        else:
            # Add zeros for channels not present in the AntiD file
            spectra_dict[v] = [0.0] * len(group_df)

    PyDict["antideuterons"][str(mdm)] = spectra_dict

# --- Save dictionary ---
FILENAME = os.path.join("PPPC4DMID", "CosmiXs.npy")
if os.path.isfile(FILENAME):
    os.remove(FILENAME)
np.save(FILENAME, PyDict)
