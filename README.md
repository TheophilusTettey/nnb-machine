# N&B Machine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21284195.svg)](https://doi.org/10.5281/zenodo.21284195)

© 2026 Theophilus T. Tettey

Open-source software for automated **Number & Brightness** analysis of Zeiss `.lsm` movies (movie QC → I–B gating → per-cell **B** and **ε**).

**Scope:** Nucleus + bright nuclear punctum N&B (SimFCS/Excel-style workflow). Not a general cell-segmentation tool — retrain QC for new cell types.

**Validation** — checked against **three** published references:

| Reference | What was checked |
|-----------|------------------|
| Jiménez-Panizo *et al.* (2022) [*Nucleic Acids Res.* **50**(22):13063–13082](https://doi.org/10.1093/nar/gkac1119) | Per-cell B/ε vs expert manual SimFCS + Excel analysis |
| Presman *et al.* (2016) [*Proc. Natl. Acad. Sci. USA* **113**, E3351–E3360](https://doi.org/10.1073/pnas.1606774113) | Published nucleus/MMTV-array oligomerization trends |
| Nolan *et al.* (2017) [*Bioinformatics* **33**, 678–680](https://doi.org/10.1093/bioinformatics/btx434) | Simulated `nandb` stacks — per-pixel B vs reference CSVs |

N&B theory (Digman *et al.* 2008) and the live-cell workflow this software automates (Presman *et al.* 2014) are cited under **Methods** in [REFERENCES.md](REFERENCES.md).

**Setup:** No interactive prompts. Edit `nnb_pipeline/config.yaml`, then run commands from this folder. Data can live on any disk; outputs go to `nnb_analysis_output/machine/`.

---

## Install

```bash
git clone https://github.com/TheophilusTettey/nnb-machine.git
cd nnb-machine
pip3 install -r nnb_pipeline/requirements.txt
cp config.example.yaml nnb_pipeline/config.yaml
```

Python **3.10+**. Mac, Linux, or Windows.

---

## Your data layout

```
raw_data_root/                         ← set in config.yaml
  MY_SESSION/                          ← one imaging day / run
    monomer_folder/                      ← control (any folder name you choose)
      01-anything.lsm
      02-anything.lsm
    mutant_A_folder/
      01-anything.lsm
    mutant_B_folder/
      01-anything.lsm
```

**File names:** must **start with a cell number** (`01-…lsm`, `02-…lsm`). The rest of the name is yours — match files to conditions with `file_glob` in config.

**Several mutants:** add one `conditions` entry per mutant (and one for monomer). No limit in code.

---

## Configure (`nnb_pipeline/config.yaml`)

| Key | Purpose |
|-----|---------|
| `raw_data_root` | Top folder containing session directories |
| `sessions.MY_SESSION.folder` | Session folder name under `raw_data_root` |
| `sessions.MY_SESSION.conditions` | List of control + mutants (`id`, `subdir`, `file_glob`) |
| `sessions.MY_SESSION.control_condition` | **Optional** — monomer condition `id` for normalized summary only |
| `manual_workbooks.MY_SESSION` | Excel labels for QC training (first time only) |

Example — monomer + two mutants:

```yaml
raw_data_root: "/path/to/microscopy"

manual_workbooks:
  MY_SESSION: "/path/to/reference-workbook.xlsx"

sessions:
  MY_SESSION:
    folder: "2024-03-15-NnB"
    control_condition: MONOMER    # omit if you don't want norm_* columns
    conditions:
      - id: MONOMER
        subdir: monomer_transfected
        file_glob: "*GRmon*.lsm"
        manual_sheet_block: "GRmon Nuc"
      - id: MUTANT_A
        subdir: mutant_A_stable
        file_glob: "*MutA*.lsm"
        manual_sheet_block: "MutantA Nuc"
      - id: MUTANT_B
        subdir: mutant_B_stable
        file_glob: "*MutB*.lsm"
        manual_sheet_block: "MutantB Nuc"
```

The program does **not** search for fixed mutant or control names — only what you define in config.

**Monomer normalization:** not automatic. If `control_condition` is set, `summary_*.csv` adds `norm_nucleus` / `norm_array` (mean ε ÷ monomer mean ε). Per-cell `cells_*.csv` is always raw B and ε.

See [`config.example.yaml`](config.example.yaml) for a full template.

---

## Run

**First time** (labeled workbook for QC):

```bash
python3 train_qc_model.py --sessions MY_SESSION
python3 fit_ib_windows.py --session MY_SESSION      # recommended
python3 run_nnb_machine.py --session MY_SESSION
```

**Repeat runs:**

```bash
python3 run_nnb_machine.py --session MY_SESSION
```

**All-in-one:**

```bash
python3 run_nnb_machine.py --session MY_SESSION --train-qc --fit-windows
```

**Subset of conditions:**

```bash
python3 run_nnb_machine.py --session MY_SESSION --conditions MONOMER MUTANT_A
```

---

## Outputs

| File | Contents |
|------|----------|
| `cells_MY_SESSION.csv` | Per-cell B, ε, QC (`condition` column per mutant) |
| `summary_MY_SESSION.csv` | Mean ε per condition; optional `norm_*` if `control_condition` set |
| `review_queue_*.csv` | Borderline cells for manual check (optional) |

---

## Cite

Theophilus T. Tettey (2026) N&B Machine. https://doi.org/10.5281/zenodo.21284195  
GitHub: https://github.com/TheophilusTettey/nnb-machine
See [REFERENCES.md](REFERENCES.md) and [CITATION.cff](CITATION.cff).

MIT License — [LICENSE](LICENSE)