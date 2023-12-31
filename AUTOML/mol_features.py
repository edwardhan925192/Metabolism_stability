from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import EState
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors, GraphDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from rdkit import Chem

def get_mol_features(mol):

    atomic_nums = 0
    total_valences = 0
    degrees = 0
    hybridizations = 0
    aromatic_count = 0
    in_ring_count = 0
    double_bond_count = 0

    for atom in mol.GetAtoms():
        atomic_nums += atom.GetAtomicNum()
        total_valences += atom.GetTotalValence()
        degrees += atom.GetDegree()
        hybridizations += int(atom.GetHybridization())
        aromatic_count += int(atom.GetIsAromatic())
        in_ring_count += int(atom.IsInRing())
        double_bond_count += sum(1 for _ in atom.GetBonds() if _.GetBondType() == Chem.BondType.DOUBLE)

    features = [
        atomic_nums,
        total_valences,
        degrees,
        hybridizations,
        aromatic_count,
        in_ring_count,
        double_bond_count
    ]

    return features

def data_prep(data):
  atom_stereo_centers = []
  F_counts, Cl_counts, Br_counts = [], [], []
  estate_indices = []
  chi0vs = []
  kappa1s = []
  complexities = []
  hk_alphas = []
  diameters = []

  matches = {}

  mol_features_list = []

  for simile in data['SMILES']:
      mol = Chem.MolFromSmiles(simile)
      if not mol:  # Check if molecule is valid
          continue

      functional_groups = {
          "hydroxyl": "[OH]",
          "amines": "[NX3;H2,H1;!$(NC=O)]",
          "carboxylic_acids": "C(=O)[O;H1]",
          "epoxides": "O1CC1",
          "alkyl_groups": "[CX4]",              
          }

      matches = {}

      # Check how many times each functional group appears in the molecule
      for group, smarts in functional_groups.items():
          pattern = Chem.MolFromSmarts(smarts)
          match_count = len(mol.GetSubstructMatches(pattern))
          matches[group] = match_count

      mol_features = get_mol_features(mol) if get_mol_features else []
      mol_features.extend(matches.values())  # Add functional group counts to mol_features

      mol_features_list.append(mol_features)

      # Storing bond connections
      one_connection = []
      for bond in mol.GetBonds():
          begin_atom_idx = bond.GetBeginAtomIdx()
          end_atom_idx = bond.GetEndAtomIdx()
          one_connection.append((begin_atom_idx, end_atom_idx))

      if (mol.GetNumAtoms() - mol.GetNumHeavyAtoms()) == 0:
          fraction = 1  # 100% heavy atoms
      else:
          fraction = mol.GetNumHeavyAtoms() / (mol.GetNumAtoms() - mol.GetNumHeavyAtoms())

      num_stereocenters = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
      atom_stereo_centers.append(num_stereocenters)

      F_counts.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[F]"))))
      Cl_counts.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[Cl]"))))
      Br_counts.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[Br]"))))

      estate_indices.append(EState.EStateIndices(mol)[0])

      chi0vs.append(GraphDescriptors.Chi0v(mol))
      kappa1s.append(GraphDescriptors.Kappa1(mol))
      complexities.append(GraphDescriptors.BertzCT(mol))

      hk_alpha = rdMolDescriptors.CalcHallKierAlpha(mol)
      hk_alphas.append(hk_alpha)

      diameter = Chem.rdMolDescriptors.CalcLabuteASA(mol)
      diameters.append(diameter)

  # Convert extracted data into DataFrame
  new_data = {
      "AtomStereoCenters": atom_stereo_centers,
      "F_counts": F_counts,
      "Cl_counts": Cl_counts,
      "Br_counts": Br_counts,
      "EStateIndices": estate_indices,
      "Chi0vs": chi0vs,
      "Kappa1s": kappa1s,
      "Complexities": complexities,
      "HK_alphas": hk_alphas,
      "Diameters": diameters,
  }

  mol_features_cols = [
  "AtomicNums",
  "TotalValences",
  "Degrees",
  "Hybridizations",
  "AromaticCount",
  "InRingCount",
  "DoubleBondCount"
  ]

  mol_features_cols.extend(functional_groups.keys())

  mol_features_df = pd.DataFrame(mol_features_list, columns=mol_features_cols)
  new_data_df = pd.DataFrame(new_data)

  result_df = pd.concat([data, new_data_df, mol_features_df], axis=1)

  return result_df

