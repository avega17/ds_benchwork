from osgeo import gdal
import gdaltools 
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib.request
import time
import os
from glob import glob
import subprocess
import pandas as pd
import multiprocessing as mp
import timeit

FILE_FMT_MAP = {"geojson":".geojson", "shapefile":".shp", "flatgeobuf":".fgb", "geopackage":".gpkg"}
FMT_GDAL_DRIVERS = {"geojson":"GeoJSON", "shapefile":"ESRI Shapefile", "flatgeobuf":"FlatGeobuf", "geopackage":"GPKG"}

def download_with_progress(url, local_file_path):
    response = urllib.request.urlopen(url)
    total_size = int(response.getheader('Content-Length').strip())
    block_size = 1024  # 1 Kibibyte
    
    print(f"Downloading {url} to {local_file_path}")

    with open(local_file_path, 'wb') as file, tqdm(
        desc=local_file_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            file.write(buffer)
            bar.update(len(buffer))

def fetch_geoparquet(country_code, output_dir):
    
    # URI for combined open buildings data
    combined_open_buildings_uri = "https://data.source.coop/vida/google-microsoft-open-buildings/geoparquet/by_country"
    # fetch geoparquet from source.coop
    url = f"{combined_open_buildings_uri}/country_iso={country_code}/{country_code}.parquet"
    output_file = os.path.join(output_dir, f"{country_code}.parquet")
    
    # verify if file already exists then return path to file
    if os.path.exists(output_file):
        print(f"{country_code}.parquet already exists at {output_file}")
        return output_file
    else: 
        t1 = time.time()
        download_with_progress(url, output_file)
        t2 = time.time()
        print(f"Succesfully downloaded {country_code}.parquet in {t2-t1} seconds and saved to {output_file}")
        return output_file

def load_geodataframe(input_file, country_code, output_format="parquet", test_load=False):

    load_time = None
    # load geopandas dataframe from geoparquet
    if output_format == "parquet":
        # use timeit to get TimeItResult object and extract median of runs 
        if test_load:
            print(f"Testing load time for {country_code}.parquet")
             # create a Timer object to measure the time taken by the function
            timer = timeit.Timer(lambda: gpd.read_parquet(input_file))
            # save median of all_runs
            load_time = np.median(timer.repeat(repeat=1, number=1))
        input_df = gpd.read_parquet(input_file)
    else:
        if test_load:
            print(f"Testing load time for {input_file.split(os.sep)[-1]}")
             # create a Timer object to measure the time taken by the function
            timer = timeit.Timer(lambda: gpd.read_file(input_file))
            # save median of all_runs
            load_time = np.median(timer.repeat(repeat=1, number=1))
            
        input_df = gpd.read_file(input_file)
    input_df.name = country_code
    print(f"Successfully loaded {country_code}.parquet into geopandas dataframe")
    
    return input_df, load_time

def pretty_print_df_info(input_df):
    print(f"Dataframe info for {input_df.name}:")
    print(input_df.info())
    print(input_df.head())

def get_output_path(input_file, output_format):
    global FILE_FMT_MAP
    # output path is same as input, but up two 
    # input has form /path/to/data/format/country_code.format
    country_code = input_file.split(os.sep)[-1].split(".")[0] # use same country code for output file name
    output_file = os.sep.join(input_file.split(os.sep)[:-2]) 
    # output file format is /path/to/data/format/country_code.format
    output_file = f"{output_file}/{output_format}/{country_code}{FILE_FMT_MAP[output_format]}"
    return output_file

# use ogr2ogr to convert geoparquet to our target formats; delete output file and return time taken and file size
def ogr_gdal_convert(input_file, output_format, delete_output=True, test_load=False, multi_proc=False):

    output_file = get_output_path(input_file, output_format)
    convert_time = time.time()
    file_size = 0
    load_time = None
    # get gdal format name
    input_format = input_file.split(".")[-1] # get file extension
    print(f"Starting conversion from {input_format} to {output_format}")
    gdal_format = FMT_GDAL_DRIVERS[output_format]
    
    if multi_proc:
        # get path of anaconda's gdal ogr2ogr with subprocess by capturing output of which ogr2ogr
        ogr_path = subprocess.run(["which", "ogr2ogr"], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
        # use pygdaltools to avoid issues with gdal parquet driver not found in spawned processes
        ogr = gdaltools.ogr2ogr(command_path=ogr_path)\
            .set_input(input_file)\
            .set_output(output_file)
        ogr.execute()
    else:
        # use subprocess to run ogr2ogr command
        command = ["ogr2ogr", "-f", gdal_format, output_file, input_file]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t2 = time.time()
    convert_time = t2 - convert_time
    
    # verify conversion result
    if result.returncode == 0:
        # calculate file size of converted file and convert to MB
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Successfully converted {input_file.split(os.sep)[-1]} to {output_file.split(os.sep)[-1]} in {convert_time:.2f}s")
        print(f"Converted file size: {file_size:.2f} MB")
        
    else:
        print(f"Error in conversion: {result.stderr.decode()}")
        return None, None, None
    
    if test_load:
        # load geopandas dataframe from converted file
        country_code = input_file.split(os.sep)[-1].split(".")[0] 
        _, load_time = load_geodataframe(input_file=output_file, country_code=country_code, output_format=output_format, test_load=True)

    # delete output file
    if delete_output:
        # delete other files like in the case of shapefile
        for file in glob(f"{output_file.split('.')[0]}*"):
            os.remove(file)
        
    return convert_time, file_size, load_time

# use geopandas to save to our target formats from existing dataframe; delete output file and return time taken and file size
def geopandas_convert(input_df, data_dir, output_format, delete_output=True):
    
    global FILE_FMT_MAP
    # get country_code from input_df
    output_file = f"{data_dir}/{output_format}/{input_df.name}{FILE_FMT_MAP[output_format]}"
    convert_time = time.time()
    load_time = None
    
    try:
        input_df.to_file(output_file, driver=FMT_GDAL_DRIVERS[output_format])
        t2 = time.time()
        convert_time = t2 - convert_time
    except Exception as e:
        print(f"Error converting {input_df} to {output_format}: {e}")
        return None, None

    # calculate file size of converted file and convert to MB
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    # delete output file
    if delete_output:
        # delete other files like in the case of shapefile
        for file in glob(f"{output_file.split('.')[0]}*"):
            os.remove(file) 
            
    print(f"Successfully converted {input_df.name} gdf to {output_file.split(os.sep)[-1]} in {convert_time:.2f}s")
    print(f"{output_format} file size: {file_size:.2f} MB")

    return convert_time, file_size

# use duckdb to read in geoparquet and save to our target formats
def duckdb_convert(input_file, output_format):
    output_file = get_output_path(input_file, output_format)

# receive geopandas df, save file to disk using specified algorithm, delete output file and return time taken and file size
def gdf_to_compressed_geoparquet(input_df, data_dir, compression_type, test_load=False):
    
    output_file = os.path.join(data_dir, "parquet", f"{input_df.name}_{compression_type}.parquet")
    convert_time = time.time()
    load_time = None

    try:
        input_df.to_parquet(output_file, compression=compression_type, schema_version='1.0.0')
        t2 = time.time()
        convert_time = t2 - convert_time
    except Exception as e:
        print(f"Error saving {input_df.name} gdf to geoparquet compressed with {compression_type}: {e}")
    
    # calculate file size of converted file and convert to MB
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    
    if test_load:
        print(f"Testing load time for {input_df.name}_{compression_type}.parquet")
        # load geopandas dataframe from converted file
        _, load_time = load_geodataframe(input_file=output_file, country_code=input_df.name, output_format="parquet", test_load=True)
    
    
    # delete output file
    os.remove(output_file)
    print(f"Successfully saved {input_df.name} gdf to geoparquet compressed with {compression_type} in {convert_time} seconds.")
    print(f"Converted file size: {file_size} MB")    

    return convert_time, file_size, load_time

# benchmark performance of converting to other vector formats 

def convert_benchmark(country_code, file_formats, data_dir, delete_output, test_load):
    
    # initialize stats for conversion
    convert_stats = {
            "processing_time": [], # for different conversion methods (ogr2ogr, geopandas, duckdb)
            "file_size": 0.0, 
            "load_time": 0.0 # time taken to load converted file into geopandas dataframe
    }
    conversion_stats = {}    
        
    # download geoparquet for each benchmarked country if not already downloaded
    input_file = os.path.join(data_dir, "parquet", f"{country_code}.parquet")
    if not os.path.exists(input_file):
        print(f"{country_code}.parquet not found, fetching from source.coop...")
        input_file = fetch_geoparquet(country_code, os.path.join(data_dir, "parquet"))
    else:
        print(f"{country_code}.parquet found at {input_file}")
    conversion_stats[country_code] = {}

    # calculate processing time and file size for geoparquet 
    parquet_file_size = os.path.getsize(input_file) / (1024 * 1024)
    country_gdf, load_time = load_geodataframe(input_file, country_code, output_format="parquet", test_load=test_load)
    conversion_stats[country_code]["parquet"] = {"file_size": parquet_file_size, "processing_time": [0.0, 0.0], "load_time": load_time}
    
    for output_format in file_formats:
        # create output dir for each file format if it doesn't exist
        output_dir = f"{data_dir}/{output_format}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # initialize stats for country_code and output_format
        conversion_stats[country_code][output_format] = convert_stats.copy() 
        # gdal/ogr2ogr conversion
        ogr_time, ogr_size, load_time = ogr_gdal_convert(input_file, output_format, delete_output, test_load)
        # geopandas conversion
        gpd_time, gpd_size = geopandas_convert(country_gdf, data_dir, output_format, delete_output)
        # duckdb conversion
        # TODO: implement duckdb conversion
        
        # record stats
        conversion_stats[country_code][output_format]["processing_time"] = [ogr_time, gpd_time]
        # file size for a given format should be mostly the same, so we take the max
        conversion_stats[country_code][output_format]["file_size"] = max(ogr_size, gpd_size)
        conversion_stats[country_code][output_format]["load_time"] = load_time
        # format to two decimal places
        print(f"file sizes for {output_format} - ogr/gdal:{ogr_size:.2f} MB, geopandas:{gpd_size:.2f} MB")
    
    return conversion_stats
        
 # benchmark compression performance of geoparquet
 
def compress_benchmark(country_code, compression_types, data_dir, delete_output, test_load):
     
    # initialize stats for compression
    compress_stats = {
                        "compression_size": 0.0,
                        "compression_time": 0.0,
                        "load_time": 0.0, 
                        "geom_count": 0
    }
    compression_stats = {}
    
    # download geoparquet for each benchmarked country if not already downloaded
    input_file = os.path.join(data_dir, "parquet", f"{country_code}.parquet")
    if not os.path.exists(input_file):
        print(f"{country_code}.parquet not found, fetching from source.coop...")
        input_file = fetch_geoparquet(country_code, os.path.join(data_dir, "parquet"))
    else:
        print(f"{country_code}.parquet found at {input_file}")

    country_gdf, _ = load_geodataframe(input_file, country_code)
    geom_count = len(country_gdf)
    compression_stats[country_code] = {}

    # compress geopandas dataframe
    for ctype in compression_types:
        
        ctype = "None" if ctype is None else ctype
        compression_stats[country_code][ctype] = compress_stats.copy()
        compress_time, compress_size, load_time = gdf_to_compressed_geoparquet(country_gdf, data_dir, ctype, test_load=test_load)
        compression_stats[country_code][ctype]["compression_time"] = compress_time
        compression_stats[country_code][ctype]["compression_size"] = compress_size
        compression_stats[country_code][ctype]["load_time"] = load_time
        compression_stats[country_code][ctype]["geom_count"] = geom_count
        
    return compression_stats

def flatten_benchmark_stats(benchmark_stats, column_name, stats_names):
    
    # both dicts have the same two levels of keys (country_code and output_format/compression_type)
    # flatten dict into single level of columns
    data = []
    for country_code, country_stats in benchmark_stats.items():
        for key, stats in country_stats.items(): 
            row = {"country_code": country_code, column_name: key}
            for stat_name in stats_names:
                row[stat_name] = stats[stat_name]  # Fix: Access stats using the 'stats' variable instead of 'benchmark_stats'][stat_name]
            data.append(row)
    
    df = pd.DataFrame(data)
    return df

def save_benchmark_stats(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Saved benchmark stats with {len(df)} records to {output_file} ({os.path.getsize(output_file) / 1024:.2f} KB)")
    
    
def init_worker():
    # Set the GDAL environment variables
    os.environ['CONDA_PREFIX'] = '/Volumes/Expanse/mambaforge/envs/geospatial'
    os.environ['GDAL_DRIVER_PATH'] = '/Volumes/Expanse/mambaforge/envs/geospatial/lib/gdalplugins'
    # add prev to LD_LIBRARY_PATH 
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib:{os.environ['CONDA_PREFIX']}/lib/gdalplugins:{os.environ['LD_LIBRARY_PATH']}"
    os.environ['GDAL_DATA'] = '/Volumes/Expanse/mambaforge/envs/geospatial/share/gdal'

    # Explicitly register GDAL drivers
    gdal.UseExceptions()
    gdal.AllRegister()

# full benchmarking pipeline using multiprocessing over the country_list 
def full_benchmark(country_list, file_formats, compression_types, data_dir, delete_output, test_load, env_vars=None, multi_proc=False, save_results=True):
    
    conversion_stats = {}
    compression_stats = {}
    
    if multi_proc:
        # process each country in parallel using multiprocessing Pool
        proc_count = mp.cpu_count() - 1
        # use multiprocessing to benchmark conversion performance for each country
        with mp.Pool(proc_count) as pool:
            conversion_stats = pool.starmap(convert_benchmark, [(country, file_formats, data_dir, delete_output, test_load) for country in country_list])
        # go through results list, flatten and concatenate into single dataframe, then save to csv 
        convert_df = pd.concat([flatten_benchmark_stats(stats, "output_format", ["processing_time", "file_size", "load_time"]) for stats in conversion_stats])
        
        # use multiprocessing to benchmark compression performance for each country
        with mp.Pool(proc_count) as pool:
            compression_stats = pool.starmap(compress_benchmark, [(country, compression_types, data_dir, delete_output, test_load) for country in country_list])
        # go through results list, flatten and concatenate into single dataframe, then save to csv
        compress_df = pd.concat([flatten_benchmark_stats(stats, "compression_type", ["compression_time", "compression_size", "load_time", "geom_count"]) for stats in compression_stats])
        
    else:
        # we'll be updating our stats dict each iteration before flattening and saving to csv
        for country_code in tqdm(country_list, desc="Benchmarking countries"):
            print(f"Testing conversion performance for {country_code}...")
            conversion_stats[country_code] = convert_benchmark(country_code, file_formats, data_dir, delete_output, test_load)
            print(f"Conversion benchmark for {country_code} complete.")
            
            print(f"Testing compression performance for {country_code}...")
            compression_stats = compress_benchmark(country_code, compression_types, data_dir, delete_output, test_load)
            print(f"Compression benchmark for {country_code} complete.")
            
        # flatten into df
        convert_df = flatten_benchmark_stats(conversion_stats, "output_format", ["processing_time", "file_size", "load_time"])
        compress_df = flatten_benchmark_stats(compression_stats, "compression_type", ["compression_time", "compression_size", "load_time", "geom_count"])

    if save_results:
        # concatenate country codes as part of filename
        convert_filename = "_".join(country_list) + "_conversion_benchmark.csv"
        save_benchmark_stats(convert_df, convert_filename)
        compress_filename = "_".join(country_list) + "_compression_benchmark.csv"
        save_benchmark_stats(compress_df, compress_filename)
    
    return convert_df, compress_df