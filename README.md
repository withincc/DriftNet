# DriftNet

Marine Drift Trajectory Prediction Model


## Project Introduction
DriftNet is a deep learning-based tool for marine drift trajectory prediction, supporting drift trajectory simulation and analysis using oceanographic forcing data.


## Environment Setup# 1. Create and activate virtual environment (recommended)
conda create -n driftnet python=3.9 -y
conda activate driftnet

# 2. Install PyTorch (compatible with CUDA 12.1)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install marine data processing dependencies
conda install -c conda-forge xarray dask netCDF4 bottleneck -y

# 4. Install mapping visualization dependencies
conda install -c conda-forge cartopy -y

# 5. Install basic visualization tools
pip install matplotlib

## Project StructureDriftNet/
├── Data/          # Contains required oceanographic forcing data
├── tools/         # Contains various research implementation tools (data processing, visualization, etc.)
├── weight/        # Contains trained model weight files
├── result/        # Output directory for results (auto-generated or manually created)
└── test.py        # Test script (generates results in result directory when run)

## Quick Start
1. Clone the repository and navigate to the project root directory
   ```bash
   git clone https://github.com/[your-username]/DriftNet.git
   cd DriftNet
   ```

2. Prepare required resources
   - Ensure the `Data` directory contains complete oceanographic forcing data
   - Ensure the `weight` directory contains pre-trained model weights

3. Run the test script
   ```bash
   python test.py
   ```

4. View results
   After completion, prediction results (trajectory data, analysis reports, etc.) will be automatically saved to the `result` directory.


## Frequently Asked Questions
- **Data missing errors**: Check if the `Data` directory contains complete forcing data, or re-download to supplement the dataset.
- **CUDA version mismatch**: Verify local CUDA version is 12.1, or replace the PyTorch installation command to match your CUDA version.
- **No output in result directory**: Check output path configuration in `test.py`, or ensure write permissions for the project directory.





