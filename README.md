# ds_benchwork
 work performed between projects to test out innovations we can adopt in the DS team

## Install dependencies
```bash
pip install -r requirements.txt
```

## Pre-requisites

- Install gdal (recommend via [anaconda](https://docs.anaconda.com/free/miniconda/)):
```bash
conda install -c conda-forge gdal
# if using conda we also need to install the geoparquet driver
conda install -c conda-forge libgdal-arrow-parquet
```
- create .env file with the following variables
```bash
DATA_DIR=/PATH/TO/OPEN/BUILDINGS/DATA
```