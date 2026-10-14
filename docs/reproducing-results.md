# Rerunning without losing the recorded results

Train in a separate checkout. The scripts use fixed model filenames, so saving a new console log alone does not prevent model replacement.

## Create an experiment checkout

From a clean, committed main clone:

```bash
git status --short
git worktree add --detach ../hybrid-run HEAD
cd ../hybrid-run
```

This checkout has separate working files and its own `model/` directory. Use a new worktree for each experiment if you want to retain each pair of models. The dataset must be present; a sparse checkout must include `DataSet` before training.

The original evidence is also stored in [original-results.zip](../results/reference/original-results.zip) with a [checksum manifest](../results/reference/manifest.json).

## Record the environment and run

Create and activate a Python environment, install `requirments.txt`, then use these commands in PowerShell from the experiment root:

```powershell
$runPath = Join-Path 'results/runs' (Get-Date -Format 'yyyy-MM-dd-HHmmss')
New-Item -ItemType Directory -Path $runPath | Out-Null
git rev-parse HEAD | Set-Content (Join-Path $runPath 'source-commit.txt')
python --version 2>&1 | Set-Content (Join-Path $runPath 'python-version.txt')
python -m pip freeze | Set-Content (Join-Path $runPath 'environment.txt')
python "codes/BaseModel(CNN+MobileV2).py" 2>&1 | Tee-Object -FilePath (Join-Path $runPath 'base-console.txt')
# Continue only after the base finishes successfully and saves its extractor.
python "codes/KNN+BaseModel(CNN+MobileV2).py" 2>&1 | Tee-Object -FilePath (Join-Path $runPath 'hybrid-console.txt')
```

Note the OS, CPU/GPU, TensorFlow device and oneDNN status in run notes. The historical dependency versions were not saved, so do not describe your new environment as the original one.

Save each confusion matrix and ROC figure to the new run directory using the plot window's save control. Close each window to allow the script to finish. Neither script automatically exports images or PDFs. Do not run hybrid after a failed base run: a leftover extractor could otherwise be mistaken for a new artifact.

After both runs finish successfully:

```powershell
Copy-Item 'model/mango_basis_extractor.h5' -Destination $runPath
Copy-Item 'model/final_hybrid_model.pkl' -Destination $runPath
Copy-Item 'class_indices.json' -Destination $runPath
Get-ChildItem -LiteralPath $runPath -File | Get-FileHash -Algorithm SHA256 |
    Select-Object Path, Hash | Export-Csv (Join-Path $runPath 'checksums.csv') -NoTypeInformation
```

The class mapping is valid only while the original class folders remain unchanged. Model files are ignored by Git; that is not a backup. Retain the experiment checkout or copy the run folder to your backup location.

New `results/runs/` folders are ignored to reduce accidental publication of models and logs. Review and explicitly add selected new artifacts if you want to publish a run. Keep the reference archive and original figures unchanged.

## Pure KNN and comparison limits

The pure KNN script requires the exact Windows dataset path in the [script guide](../codes/README.md). Changing the working directory will not make it portable. An unchanged-code run requires that dataset location. It saves no fitted model; capture its logs and plots separately if that environment is available.

Pure KNN is not stratified; base and hybrid are. A shared saved split across all models would require code changes. This documentation update leaves those differences intact and reports them explicitly.

Compare new metrics and per-class supports with [recorded-metrics.json](../results/reference/recorded-metrics.json), rather than replacing the historical numbers. Keep the extractor and hybrid PKL from the same run together.

## Why the seed is not a guarantee

All scripts use seed 42, but image enumeration uses unsorted `os.listdir`, dependency versions are broad and numerical operations can vary. `PYTHONHASHSEED` is set inside the running scripts, after interpreter startup, and the hybrid sets its TensorFlow environment flag after importing TensorFlow. Exact split file lists are not saved.

The reference archive preserves the historical evidence. It does not force future runs to produce the same numbers. No training or new evaluation was performed during this documentation update.
