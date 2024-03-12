from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

mols = Chem.SDMolSupplier('135.sdf')
mol = mols[0]
# mol = Chem.MolFromMolFile('101.mol')
mub = 0
# 获取原子特征
print(type(mol))
for atom in mol.GetAtoms():
    print(atom.GetAtomicNum(), atom.GetIdx())
    nub = atom.GetIdx()

# 获取连接建特征
bonds = mol.GetBonds()
for bond in bonds:
    print(bond.GetBondType())
    print(bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum())
    print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
# 获取位置
for i in range(mol.GetNumAtoms()):
    x, y, z = mol.GetConformer().GetAtomPosition(i)
    print(x, y, z)

# 添加H原子 3D构象优化的时候，需要采用显式H原子
m1 = Chem.AddHs(mol)
print("num ATOMs in m:",mol.GetNumAtoms())
print("num ATOMs in m1:",m1.GetNumAtoms())

# 拓扑指纹
fps = Chem.RDKFingerprint(mol)
fps1 = Chem.RDKFingerprint(m1)
print(fps) # result 1
print(len(fps.ToBitString())) # result 2
print(fps.ToBitString())
print(fps1.ToBitString())

# MACCSkeys指纹,长度为167的分子指纹
fingerprints = MACCSkeys.GenMACCSKeys(mol)
print(fingerprints) # result 1
print(len(fingerprints.ToBitString())) # result 2
print(fingerprints.ToBitString()) # result 3


# Atom Pairs
fps0 = Pairs.GetAtomPairFingerprintAsBitVect(mol)
print(fps0)
print(len(fps0.ToBitString()))
# 拓扑扭曲 GetTopologicalTorsionFingerprint(mol, targetSize, ...)
fps1 = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
print('拓扑')
print(fps1)

# Morgan Fingerprints (Circular Fingerprints)
fps2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
print(f'mogen{np.array(fps2)}')
print(fps2)
print(len(fps2.ToBitString()))
print(fps2.ToBitString())


# ECFP4  RDKit 中的Morgan算法支持feature,ECFP和FCFP中的4代表是摩根指纹的直径为4，
# 半径为2，默认半径为2的摩根指纹就是ECFP指纹，半径为2且考虑feature-based  invariants得到的指纹为FCFP4指纹。
ecfp4_mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
fcfp4_mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)
print(ecfp4_mg,fcfp4_mg)
print(len(ecfp4_mg.ToBitString()), ecfp4_mg.ToBitString())
print(len(fcfp4_mg.ToBitString()), fcfp4_mg.ToBitString())

smi = 'OC(=O)c1ccc(O)cc1'
mol = Chem.MolFromSmiles(smi)
pfingerprints = MACCSkeys.GenMACCSKeys(mol)
print(fingerprints.ToBitString())
print(len(fingerprints.ToBitString()))
# mol = Chem.MolFromMolFile('135.sdf')
# mol2 = Chem.SDMolSupplier('135.sdf')[0]
# for atom in mol.GetAtoms():
#     print(atom.GetAtomicNum(), atom.GetIdx())
#     nub = atom.GetIdx()
#
# for atom in mol2.GetAtoms():
#     print(atom.GetAtomicNum(), atom.GetIdx())
#     nub = atom.GetIdx()
