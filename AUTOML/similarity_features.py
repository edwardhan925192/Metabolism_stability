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
    mlm_max = 'O=C(Nc1ccccc1)C1CCCN1C1=NS(=O)(=O)c2ccccc21'
    mlm_min = 'COCCCNC(=O)C1CCC(CN(Cc2c(Cl)cccc2Cl)S(=O)(=O)c2ccc(Br)cc2)CC1'
    hlm_max = 'CN(C)S(=O)(=O)CCNCc1ccc(-c2ccccc2)cc1'
    hlm_min = 'COCCCNC(=O)C1CCC(CN(Cc2c(Cl)cccc2Cl)S(=O)(=O)c2ccc(Br)cc2)CC1'
    maxgap = 'CN(C)S(=O)(=O)CCNCc1ccc(-c2ccccc2)cc1'
    mingap = 'O=C(C1CC(=O)N(c2n[nH]c3cc(Br)ccc23)C1)N1CCCC1'
    
    standards = [mlm_max, mlm_min, hlm_max, hlm_min, maxgap, mingap]
    standard_names = ['mlm_max', 'mlm_min', 'hlm_max', 'hlm_min', 'maxgap', 'mingap']

    results = []
    for smiles in df['SMILES']:
        similarities = [compute_similarity(smiles, standard, nBits) for standard in standards]
        results.append(similarities)

    df_similarities = pd.DataFrame(results, columns=standard_names)

    return pd.concat([df, df_similarities], axis=1)
