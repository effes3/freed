import sys
import os
from subprocess import run, PIPE

vina_program = "/home/semakin_grisha/FFREED/utils/qvina02"
receptor = "/home/semakin_grisha/FFREED/ffreed/data/receptors/protein.pdbqt"
box_center = [17.179, 35.807, 33.795]
box_size = [20.0, 20.0, 20.0]

# Simple test molecule (Ethanol)
smile = "CCO"

print("="*60)
print(f"1. Testing obabel generation for SMILES: {smile}")
print("="*60)
obabel_cmd = f"obabel -:{smile} --gen3D -h -opdbqt -O test_ligand.pdbqt"
print(f"Running: {obabel_cmd}")

result_obabel = run(obabel_cmd.split(), capture_output=True, text=True)
print("\n--- OBABEL STDOUT ---")
print(result_obabel.stdout)
print("\n--- OBABEL STDERR ---")
print(result_obabel.stderr)
print(f"Exit code: {result_obabel.returncode}")

if not os.path.exists("test_ligand.pdbqt") or os.path.getsize("test_ligand.pdbqt") == 0:
    print("FAILED: test_ligand.pdbqt was not created or is empty.")
    sys.exit(1)

print("\n" + "="*60)
print("2. Testing qvina02 Docking")
print("="*60)
vina_cmd = (
    f"{vina_program} --receptor {receptor} --ligand test_ligand.pdbqt "
    f"--out test_out.pdbqt "
    f"--center_x {box_center[0]} --center_y {box_center[1]} --center_z {box_center[2]} "
    f"--size_x {box_size[0]} --size_y {box_size[1]} --size_z {box_size[2]} "
    f"--num_modes 1 --exhaustiveness 1 --seed 150"
)
print(f"Running: {vina_cmd}")

result_vina = run(vina_cmd.split(), capture_output=True, text=True)
print("\n--- VINA STDOUT ---")
print(result_vina.stdout)
print("\n--- VINA STDERR ---")
print(result_vina.stderr)
print(f"Exit code: {result_vina.returncode}")

if os.path.exists("test_out.pdbqt"):
    print("\nSUCCESS: test_out.pdbqt was created successfully.")
else:
    print("\nFAILED: test_out.pdbqt was not created.")

print("="*60)
