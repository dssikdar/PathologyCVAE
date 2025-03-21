# Instructions to run demo code

## Requirements

Before running the demo, ensure you have the following:

- **Python 3.9+** installed
- **Jupyter Notebook** installed (`pip install notebook`)
- **Dataset & Utility Files:** Extract the provided zip file. Make sure you see the following files in the same directory:
  - `BreastHistopathology_Small.zip` 
  - `DEMO_ConvVAE.ipynb`
  - `breast_cancer_dataset.py`
  - `environment.yml`
  - `instructions.md`

## Recommended System Setup

- **Linux** (preferred) with access to at least one **GPU** (NVIDIA recommended, CUDA supported)
- **Mac** or **Windows**: Works fine as long as you have **Python 3.9+** and Jupyter Notebook installed
- **Google Colab**: 
  - Manually upload the provided files (dataset, notebook, and utility scripts) to the workspace.
  - Change the runtime to **GPU (Google T4 recommended)**:  
    `Runtime` → `Change runtime type` → Select `GPU`
  - Run all cells in sequence.

## Steps

### 1. Install Python & Jupyter Notebook
If you haven't already installed Python and Jupyter Notebook, do so using:

```bash
pip install notebook
```
or
```conda
conda install -c conda-forge notebook
```

### 2. Running the Demo

1. **Extract the ZIP file** and navigate to the extracted folder:
   ```bash
   unzip run_demo.zip -d run_demo
   cd run_demo
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the provided .ipynb file, namely `DEMO_ConvVAE.ipynb`, in Jupyter Notebook.
4. Ensure the dataset and utility files are in the same directory as the notebook.
5. Run the first cell in the notebook to install all required dependencies.
6. Then run all remaining cells sequentially.

### 3. About the mini-dataset in the demo
The dataset, self-contained in `breast_histopathology.zip`, has 1000 breast tissue histopathology images. Of that, 900 images are non-cancerous, and 100 images are cancerous. The `zip` file, only ~5MB, is a small fraction of the original [dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) and is intended only for demonstration purposes.
