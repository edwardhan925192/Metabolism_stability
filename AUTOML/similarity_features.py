import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

def compute_similarity(smiles1, smiles2, nBits=1024):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=nBits)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

def add_similarity_features(df, nBits):
    mlm_max = df.loc[662,'SMILES'] if len(df) > 662 else None
    mlm_min = df.loc[23,'SMILES'] if len(df) > 23 else None
    hlm_max = df.loc[1584,'SMILES'] if len(df) > 1584 else None
    hlm_min = df.loc[23,'SMILES'] if len(df) > 23 else None
    maxgap = df.loc[1584,'SMILES'] if len(df) > 1584 else None
    mingap = df.loc[22,'SMILES'] if len(df) > 22 else None

    standards = [mlm_max, mlm_min, hlm_max, hlm_min, maxgap, mingap]
    standard_names = ['mlm_max', 'mlm_min', 'hlm_max', 'hlm_min', 'maxgap', 'mingap']

    results = []
    for smiles in df['SMILES']:
        similarities = [compute_similarity(smiles, standard, nBits) for standard in standards]
        results.append(similarities)

    df_similarities = pd.DataFrame(results, columns=standard_names)

    return pd.concat([df, df_similarities], axis=1)
