"""
NOAA MRMS Gauge Precipitation Analysis

This script demonstrates how to access, process, and visualize NOAA MRMS (Multi-Radar/Multi-Sensor System) 
gauge precipitation data from AWS S3.
"""

# Standard library imports
import os
import sys
import tempfile
import gzip
from datetime import datetime, timedelta

# Third-party imports
import boto3
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

# Set up boto3 client for anonymous access to public S3 bucket
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
BUCKET_NAME = 'noaa-mrms-pds'

# Base directory where the data is stored - note the 'l' in 'GaugeInflIndex'
BASE_DIR = 'CONUS/GaugeInflIndex_01H_Pass2_00.00'

def list_s3_files(bucket, base_dir, date_str):
    """List all files in the S3 bucket for a specific date.
    
    Args:
        bucket: S3 bucket name
        base_dir: Base directory in the bucket (e.g., 'CONUS/GaugeInfIndex_01H_Pass2_00.00')
        date_str: Date in YYYYMMDD format
    """
    try:
        # Construct the path to the date directory
        date_prefix = f"{base_dir}/{date_str}/"
        print(f"Searching in directory: {date_prefix}")
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=date_prefix,
            MaxKeys=1000  # Make sure we get all results
        )
        
        # Print the full response for debugging
        print("\nS3 Response:")
        print(f"Response keys: {list(response.keys())}")
        if 'KeyCount' in response:
            print(f"KeyCount: {response['KeyCount']}")
        if 'IsTruncated' in response:
            print(f"IsTruncated: {response['IsTruncated']}")
        if 'MaxKeys' in response:
            print(f"MaxKeys: {response['MaxKeys']}")
            
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            print(f"\nFound {len(files)} files:")
            for f in files[:5]:  # Show first 5 files
                print(f" - {f}")
            if len(files) > 5:
                print(f" - ... and {len(files) - 5} more")
            return files
            
        print("No 'Contents' key in response. Response keys:", response.keys())
        return []
        
    except Exception as e:
        print(f"Error listing files: {e}")
        import traceback
        traceback.print_exc()
        return []

def download_and_process_grib(bucket, key):
    """Download and process a single GRIB2 file, handling gzipped files."""
    try:
        # Create a temporary directory to store the downloaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the gzipped file
            gz_path = os.path.join(temp_dir, 'temp.grib2.gz')
            grib_path = os.path.join(temp_dir, 'temp.grib2')
            
            # Download the file
            print(f"Downloading file: {key}")
            s3_client.download_file(bucket, key, gz_path)
            
            # Decompress the file
            print("Decompressing file...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(grib_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Read the decompressed GRIB2 file with cfgrib
            print("Reading GRIB2 file...")
            try:
                # First try with default parameters
                ds = xr.open_dataset(grib_path, engine='cfgrib')
                print("Successfully read GRIB2 file with default parameters")
            except Exception as e:
                print(f"Error reading GRIB2 file with default parameters: {e}")
                # Try with specific backend kwargs
                try:
                    print("Trying with specific backend kwargs...")
                    ds = xr.open_dataset(
                        grib_path,
                        engine='cfgrib',
                        backend_kwargs={
                            'filter_by_keys': {'typeOfLevel': 'surface'},
                            'indexpath': ''
                        }
                    )
                    print("Successfully read with specific backend kwargs")
                except Exception as e2:
                    print(f"Error with specific backend kwargs: {e2}")
                    raise
            
            # Load all data into memory and close the file
            print("Loading data into memory...")
            ds_loaded = ds.load()
            ds.close()  # Close the file handle
            
            # Clean up temporary files
            try:
                if os.path.exists(gz_path):
                    os.unlink(gz_path)
                if os.path.exists(grib_path):
                    os.unlink(grib_path)
            except OSError as e:
                print(f"Warning: Error cleaning up temporary files: {e}")
            
            return ds_loaded
            
    except Exception as e:
        print(f"Error downloading or processing {key}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_precipitation(ds, output_file='precipitation_map.png'):
    """Create a precipitation map from the dataset in memory."""
    try:
        print("Creating precipitation plot...")
        
        # Extract the data variable (usually the first one)
        var_name = list(ds.data_vars.keys())[0]
        data = ds[var_name]
        
        # Print debug info
        print(f"Data variable: {var_name}")
        print(f"Data shape: {data.shape}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        
        # Handle time dimension if it exists
        if 'time' in data.dims:
            plot_data = data.isel(time=0)
        else:
            plot_data = data
            
        # Create a figure with a map background
        plt.figure(figsize=(16, 9), dpi=150)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set the extent to the CONUS region with a bit of padding
        west_lon, east_lon = -126, -65  # Slightly wider than CONUS to show context
        south_lat, north_lat = 23, 51   # Slightly taller than CONUS
        ax.set_extent([west_lon, east_lon, south_lat, north_lat], crs=ccrs.PlateCarree())
        
        # Add map features with appropriate styling
        ax.coastlines(resolution='50m', color='black', linewidth=0.8)
        ax.add_feature(cfeature.STATES, linestyle='-', edgecolor='gray', linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.8)
        
        # Add gridlines with labels
        gl = ax.gridlines(
            draw_labels=True, 
            linewidth=0.5, 
            color='gray', 
            alpha=0.5, 
            linestyle='--',
            xlocs=range(-180, 180, 10),  # Show longitude lines every 10 degrees
            ylocs=range(-90, 90, 5)      # Show latitude lines every 5 degrees
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8, 'color': 'black'}
        gl.ylabel_style = {'size': 8, 'color': 'black'}
        
        # Create title with timestamp if available
        title = 'MRMS Gauge Influence Index'
        if 'time' in ds.coords:
            # Handle different possible time formats in the dataset
            try:
                if hasattr(ds.time, 'values') and len(ds.time.values) > 0:
                    time_value = ds.time.values[0] if hasattr(ds.time.values, '__getitem__') else ds.time.values
                    time_str = pd.to_datetime(str(time_value)).strftime('%Y-%m-%d %H:%M:%S UTC')
                    title = f'{title} - {time_str}'
                else:
                    # If time exists but has no values, use current time
                    time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                    title = f'{title} - {time_str} (Current Time)'
            except Exception as time_err:
                print(f"Warning: Could not parse time: {time_err}")
                time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                title = f'{title} - {time_str} (Time Unknown)'
        
        plt.title(title, fontsize=14, pad=15, weight='bold')
        
        # Plot the data with a colorbar
        print(f"Plotting data with shape {plot_data.shape}...")
        
        # Use pcolormesh for better performance with large datasets
        plot = plot_data.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            add_colorbar=True,
            cbar_kwargs={
                'label': 'Gauge Influence Index',
                'orientation': 'vertical',
                'shrink': 0.8,
                'pad': 0.02,
                'extend': 'max',
                'aspect': 30
            },
            vmin=0,  # Start color scale at 0
            vmax=1,  # Maximum value for gauge influence index
            levels=20  # Number of color levels
        )
        
        # Add a credit/source text
        plt.figtext(
            0.5, 0.01, 
            'Data: NOAA MRMS | Processed: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ha='center',
            fontsize=8,
            color='gray'
        )
        
        # Adjust layout to make room for title and colorbar
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure with high resolution
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Successfully created plot: {os.path.abspath(output_file)}")
        except Exception as save_error:
            print(f"Error saving plot: {save_error}")
            raise
        finally:
            # Ensure all figures are closed
            plt.close('all')
            
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        # Ensure figures are closed even if there was an error
        plt.close('all')
        raise

def list_bucket_contents(bucket, prefix='', delimiter='/', max_keys=1000):
    """List all objects in the S3 bucket with the given prefix."""
    try:
        print(f"Listing contents of bucket '{bucket}' with prefix: '{prefix}'")
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimiter,
            PaginationConfig={'MaxItems': max_keys}
        ):
            # Print common prefixes (subdirectories)
            if 'CommonPrefixes' in page:
                print("\nCommon Prefixes:")
                for cp in page['CommonPrefixes']:
                    print(f" - {cp['Prefix']}")
            
            # Print objects
            if 'Contents' in page:
                print("\nObjects:")
                for obj in page['Contents']:
                    print(f" - {obj['Key']} (Size: {obj['Size']} bytes, LastModified: {obj['LastModified']})")
            
            # Print any truncated results
            if page.get('IsTruncated', False):
                print("\nResults are truncated. More items are available.")
                
    except Exception as e:
        print(f"Error listing bucket contents: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to demonstrate the workflow."""
    # First, let's see what's in the CONUS directory
    print("Listing CONUS directory to find gauge data...")
    list_bucket_contents(BUCKET_NAME, 'CONUS/', '/', 50)
    
    # Let's also check if there are any gauge-related directories
    print("\nLooking for gauge-related directories in CONUS...")
    list_bucket_contents(BUCKET_NAME, 'CONUS/Gauge', '/', 50)
    
    # Example: Analyze data for February 14, 2025
    date_to_analyze = '20250214'
    
    # List available files
    print(f"Searching for files for {date_to_analyze}...")
    files = list_s3_files(BUCKET_NAME, BASE_DIR, date_to_analyze)
    
    if not files:
        print(f"No files found for {date_to_analyze}. Exiting.")
        return
    
    print(f"Found {len(files)} files. Processing first file...")
    
    # Process the first file
    sample_file = files[0]
    ds = download_and_process_grib(BUCKET_NAME, sample_file)
    
    if ds is None:
        print("Failed to process the GRIB file. Exiting.")
        return
    
    print("\nDataset information:")
    print(ds)
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save the plot
    print("\nCreating precipitation plot...")
    try:
        output_file = os.path.join(output_dir, f'mrms_precipitation_{date_to_analyze}.png')
        plot_precipitation(ds, output_file=output_file)
        print(f"Plot saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
