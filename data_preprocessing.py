import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split

def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    atom_indices = np.random.permutation(mol.GetNumAtoms()).tolist()
    randomized_mol = Chem.RenumberAtoms(mol, atom_indices)
    return Chem.MolToSmiles(randomized_mol, canonical=False)

def augment_smiles(smiles, n_augments=5):
    augmented_smiles = set()
    while len(augmented_smiles) < n_augments:
        randomized_smiles = randomize_smiles(smiles)
        augmented_smiles.add(randomized_smiles)
    return list(augmented_smiles)

def load_and_preprocess_data(csv_path="data/clintox.csv", n_augments=5, test_size=0.3, random_state=16):
    data = pd.read_csv(csv_path)
    smiles_data = data['smiles']
    labels = data['CT_TOX']

    smiles_0 = [s for s, l in zip(smiles_data, labels) if l == 0]
    smiles_1 = [s for s, l in zip(smiles_data, labels) if l == 1]

    augmented_smiles_0 = []
    augmented_smiles_1 = []

    for smiles in smiles_0:
        augmented_smiles_0.extend(augment_smiles(smiles, n_augments=n_augments))

    for smiles in smiles_1:
        augmented_smiles_1.extend(augment_smiles(smiles, n_augments=n_augments))

    augmented_smiles_data = augmented_smiles_0 + augmented_smiles_1
    augmented_labels = [0] * len(augmented_smiles_0) + [1] * len(augmented_smiles_1)

    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        augmented_smiles_data, augmented_labels, test_size=test_size, random_state=random_state
    )

    print(f"Original data size: {len(smiles_data)}")
    print(f"Augmented data size: {len(augmented_smiles_data)}")
    print(f"Train set size: {len(train_smiles)}")
    print(f"Test set size: {len(test_smiles)}")

    return train_smiles, test_smiles, train_labels, test_labels
