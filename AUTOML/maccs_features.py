from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pandas as pd
import numpy as np

def generate_and_concatenate_MACCS_keys(df, drop_columns=None):
    # Generate MACCS keys
    maccs_keys = []
    for simile in df['SMILES']:
        mol = Chem.MolFromSmiles(simile)
        if mol:  # Check if molecule is valid
            key = MACCSkeys.GenMACCSKeys(mol)
            maccs_keys.append(key.ToBitString())  # Convert the RDKit explicit bit vector to a string
        else:
            maccs_keys.append('0'*167)  # Assuming a fixed length of 167 for MACCS keys

    # Convert MACCS keys to numpy array and then to DataFrame
    maccs_np = np.array([[int(bit) for bit in key] for key in maccs_keys])
    maccs_df = pd.DataFrame(maccs_np, columns=[f"MACCS_bit_{i+1}" for i in range(167)])

    # Drop specified columns if any
    if drop_columns:
        df = df.drop(columns=drop_columns)

    # Concatenate dataframes
    concatenated_df = pd.concat([df, maccs_df], axis=1)

    return concatenated_df
