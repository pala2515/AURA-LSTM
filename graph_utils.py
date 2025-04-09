import numpy as np
from rdkit import Chem
from rdkit.Chem import SanitizeMol, MolFromSmiles

def molecule_to_graph(smiles):
    try:
        mol = MolFromSmiles(smiles)

        if mol is None:
            # print(f"Warning: RDKit could not parse SMILES: {smiles}")
            return None, None

        # Try sanitizing, but proceed even if it fails for some structures if needed
        try:
            SanitizeMol(mol)
        except Exception as sanitize_error:
            # print(f"Warning: Sanitization failed for SMILES {smiles}: {sanitize_error}. Proceeding without sanitization.")
            pass # Decide if you want to proceed or return None here

        if mol.GetNumAtoms() == 0:
             # print(f"Warning: Mol with zero atoms for SMILES: {smiles}")
             return None, None

        node_features = []
        for atom in mol.GetAtoms():
            # Basic features: Atomic number, IsAromatic
            features = [atom.GetAtomicNum()]
            features.append(1 if atom.GetIsAromatic() else 0)
            # Add more features if needed: Degree, FormalCharge, NumHydrogens, Hybridization etc.
            # features.append(atom.GetDegree())
            # features.append(atom.GetFormalCharge())
            # features.append(atom.GetTotalNumHs())
            # features.append(int(atom.GetHybridization())) # Needs conversion
            node_features.append(features)

        adj_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            # Simple adjacency: 1 if connected, 0 otherwise
            adj_matrix[begin_idx, end_idx] = 1
            adj_matrix[end_idx, begin_idx] = 1
            # Could add bond type features here if needed

        return np.array(node_features, dtype=np.float32), adj_matrix

    except Exception as e:
        # print(f"Error processing SMILES to graph: {smiles}, Error: {str(e)}")
        return None, None
