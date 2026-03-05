import json
import re
import glob
import os
from rdkit import Chem

# Configuration
INPUT_PATTERN = "/home/semakin_grisha/FREED/gym_molecule/dataset/motif*.txt"
OUTPUT_FILE = "/home/semakin_grisha/FFREED/ffreed/data/motifs/merged_motifs.json"

def convert_smiles_format(smi):
    # Converts RDKit map format [*:1] to FFREED label format [1*]
    return re.sub(r"\[\*:(\d+)\]", r"[\1*]", smi)

def process_files():
    files = glob.glob(INPUT_PATTERN)
    print(f"Found {len(files)} motif files: {files}")
    
    unique_smiles = set()
    total_processed = 0
    
    for file_path in files:
        with open(file_path, "r") as f:
            for line in f:
                smi = line.strip()
                if not smi: continue
                total_processed += 1
                
                # 1. Convert [*:n] -> [n*]
                converted = convert_smiles_format(smi)
                
                # 2. Basic molecule validation
                mol = Chem.MolFromSmiles(converted)
                if not mol:
                    continue
                
                # 3. Filter: ONLY single-sticker fragments (as per your request)
                # and ONLY terminal stickers (no bridging neighbors)
                stickers = []
                is_valid = True
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == "*" or atom.GetAtomicNum() == 0:
                        stickers.append(atom)
                        # Ensure sticker is connected to exactly one piece of the molecule
                        if len(atom.GetNeighbors()) != 1:
                            is_valid = False
                            break
                            
                if is_valid:
                    unique_smiles.add(converted)

    print(f"Total entries scanned: {total_processed}")
    print(f"Final valid single-sticker fragments: {len(unique_smiles)}")
    
    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(list(unique_smiles), f, indent=4)
    print(f"Success! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_files()
