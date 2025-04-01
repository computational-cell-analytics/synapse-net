import json
import os

root = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse"
with open("tomo_issues.json", "r") as f:
    tomos = json.load(f)

for name in tomos:
    path = os.path.join(root, name, "Korrektur", "measurements.xlsx")
    if not os.path.exists(path):
        path = os.path.join(root, name, "korrektur", "measurements.xlsx")
    if os.path.exists(path):
        print("Removing", path)
        os.remove(path)
    else:
        print("Skipping", path)
