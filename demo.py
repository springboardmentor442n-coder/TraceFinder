#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

SRC = r"D:\Infosys_AI-Tracefinder\Data\Residuals_vis"  # change if needed
rows = []

for scanner in os.listdir(SRC):
    scanner_path = os.path.join(SRC, scanner)
    if not os.path.isdir(scanner_path): continue
    for dpi in os.listdir(scanner_path):
        dpi_path = os.path.join(scanner_path, dpi)
        if not os.path.isdir(dpi_path): continue
        for f in os.listdir(dpi_path):
            if f.endswith("_res_vis.png"):  # take only residual images
                path = os.path.join(dpi_path, f)
                rows.append({
                    "path": path,
                    "scanner": scanner,
                    "dpi": dpi
                })

df = pd.DataFrame(rows)
df.to_csv(r"D:\Infosys_AI-Tracefinder\Output\residual_list.csv", index=False)
print("✅ CSV created:", len(df), "images found")

