import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
import os

def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles # Return original if RDKit can't parse it
    try:
        atom_indices = np.random.permutation(mol.GetNumAtoms()).tolist()
        randomized_mol = Chem.RenumberAtoms(mol, atom_indices)
        return Chem.MolToSmiles(randomized_mol, canonical=False)
    except Exception:
        # If renumbering fails for some reason, return original
        return smiles


def augment_smiles(smiles, n_augments=5):
    augmented_smiles = set([smiles]) # Start with the original smiles
    attempts = 0
    max_attempts = n_augments * 5 # Limit attempts to prevent infinite loops

    while len(augmented_smiles) < n_augments + 1 and attempts < max_attempts:
        randomized = randomize_smiles(smiles)
        augmented_smiles.add(randomized)
        attempts += 1

    # If we couldn't generate enough unique ones, just return what we have
    final_list = list(augmented_smiles)
    # Ensure we return exactly n_augments + 1 items if possible, otherwise fewer
    return final_list[:n_augments + 1]


def load_and_preprocess_data(datasets_info, n_augments=5, test_size=0.3, random_state=16):
    all_smiles = []
    all_labels = []
    original_total_size = 0

    print("Loading data from datasets:")
    for info in datasets_info:
        path = info['path']
        smiles_col = info['smiles_col']
        label_col = info['label_col']

        if not os.path.exists(path):
            print(f"Warning: Dataset file not found at {path}. Skipping.")
            continue

        try:
            data = pd.read_csv(path)
            print(f"- Loaded {path} ({len(data)} samples)")
            smiles_data = data[smiles_col].astype(str).tolist() # Ensure SMILES are strings
            labels = data[label_col].tolist()

            # Basic validation: check lengths match
            if len(smiles_data) != len(labels):
                 print(f"Warning: SMILES and label count mismatch in {path}. Skipping this file.")
                 continue

            all_smiles.extend(smiles_data)
            all_labels.extend(labels)
            original_total_size += len(data)

        except FileNotFoundError:
            print(f"Error: File not found {path}")
        except KeyError as e:
            print(f"Error: Column {e} not found in {path}")
        except Exception as e:
            print(f"An error occurred while processing {path}: {e}")

    if not all_smiles:
         raise ValueError("No valid data loaded. Please check dataset paths and column names.")


    # Ensure labels are integers (0 or 1 typically)
    try:
        all_labels = [int(l) for l in all_labels]
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert all labels to integers: {e}. Check label columns.")
        # Depending on the error, you might want to filter out invalid labels or raise an error


    smiles_0 = [s for s, l in zip(all_smiles, all_labels) if l == 0]
    smiles_1 = [s for s, l in zip(all_smiles, all_labels) if l == 1]

    # Note: If you have more than 2 classes, this augmentation logic needs adjustment
    if len(set(all_labels)) > 2:
        print("Warning: More than two classes detected. Augmentation logic currently assumes binary classification (0 and 1).")


    augmented_smiles_list = []
    augmented_labels_list = []

    print(f"\nAugmenting data (target: {n_augments} variations per SMILES)...")

    # Process label 0
    print("Augmenting class 0...")
    count_0 = 0
    for smiles in smiles_0:
        augmented = augment_smiles(smiles, n_augments=n_augments)
        augmented_smiles_list.extend(augmented)
        augmented_labels_list.extend([0] * len(augmented))
        count_0 += 1
        if count_0 % 500 == 0: print(f"  Processed {count_0}/{len(smiles_0)} for class 0")


    # Process label 1
    print("Augmenting class 1...")
    count_1 = 0
    for smiles in smiles_1:
        augmented = augment_smiles(smiles, n_augments=n_augments)
        augmented_smiles_list.extend(augmented)
        augmented_labels_list.extend([1] * len(augmented))
        count_1 += 1
        if count_1 % 500 == 0: print(f"  Processed {count_1}/{len(smiles_1)} for class 1")

    # Combine potentially remaining original SMILES if not processed above
    # This part might be redundant if augment_smiles includes the original
    # Let's ensure augmented_smiles includes original
    # augmented_smiles_data = augmented_smiles_0 + augmented_smiles_1 # Original code logic
    # augmented_labels = [0] * len(augmented_smiles_0) + [1] * len(augmented_smiles_1) # Original code logic

    augmented_smiles_data = augmented_smiles_list
    augmented_labels = augmented_labels_list


    if not augmented_smiles_data:
        raise ValueError("Augmentation resulted in empty dataset.")

    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        augmented_smiles_data, augmented_labels, test_size=test_size, random_state=random_state, stratify=augmented_labels # Added stratify
    )

    print(f"\nOriginal total data size: {original_total_size}")
    print(f"Augmented data size: {len(augmented_smiles_data)}")
    print(f"Train set size: {len(train_smiles)}")
    print(f"Test set size: {len(test_smiles)}")

    return train_smiles, test_smiles, train_labels, test_labels
