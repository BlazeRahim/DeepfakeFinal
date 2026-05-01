import nbformat

try:
    nb = nbformat.read(r'c:\Users\blaze\Desktop\DeepfakeFinal\VeriLens_Final.ipynb', as_version=4)

    # Find the cell currently containing !pip install tensorflow
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and '!pip install tensorflow' in cell.source:
            cell.source = "# !pip install tensorflow" 
            print(f"Commented out !pip install tensorflow in cell {i}")
            break

    nbformat.write(nb, r'c:\Users\blaze\Desktop\DeepfakeFinal\VeriLens_Final.ipynb')
    print("Successfully rebuilt the notebook!")

except Exception as e:
    print(f"Error: {e}")
