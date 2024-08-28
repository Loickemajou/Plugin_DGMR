import numpy as np
import os
import pandas as pd
import pyproj
from wradlib.io import read_opera_hdf5
import xarray as xr
import tensorflow as tf
import pysteps
from dgmr_module_plugin.dgmr import forecast
from pysteps.visualization import animations
from pysteps import datasets





def read(data_folder):
    '''Code by Simon De Kock <simon.de.kock@vub.be>'''
    fns = []
    # A slice of the files was selected to produce nowcasts with DGMR and LDCast
    # Such that those nowcast start as close as possible to the startime of the PySTEPS and INCA nowcasts
    for filename in os.listdir(data_folder)[:22]:
        if filename.endswith('hdf'):
            fns.append(f"{data_folder}/{filename}")
    
    dataset = []
    for i, file_name in enumerate(fns):
        # Read the content
        file_content = read_opera_hdf5(file_name)

        # Extract time information
        time_str = os.path.splitext(os.path.basename(file_name))[0].split('.', 1)[0]
        time = pd.to_datetime(time_str, format='%Y%m%d%H%M%S')

        # Extract quantity information
        try:
            quantity = file_content['dataset1/data1/what']['quantity'].decode()
        except:
            quantity = file_content['dataset1/data1/what']['quantity']

        # Set variable properties based on quantity
        if quantity == 'RATE':
            short_name = 'precip_intensity'
            long_name = 'instantaneous precipitation rate'
            units = 'mm h-1'
        else:
            raise Exception(f"Quantity {quantity} not yet implemented.")

        # Create the grid
        projection = file_content.get("where", {}).get("projdef", "")
        if type(projection) is not str:
            projection = projection.decode("UTF-8")

        gridspec = file_content.get("dataset1/where", {})

        x = np.linspace(gridspec.get('UL_x', 0),
                        gridspec.get('UL_x', 0) + gridspec.get('xsize', 0) * gridspec.get('xscale', 0),
                        num=gridspec.get('xsize', 0), endpoint=False)
        x += gridspec.get('xscale', 0)
        y = np.linspace(gridspec.get('UL_y', 0),
                        gridspec.get('UL_y', 0) - gridspec.get('ysize', 0) * gridspec.get('yscale', 0),
                        num=gridspec.get('ysize', 0), endpoint=False)
        y -= gridspec.get('yscale', 0) / 2

        x_2d, y_2d = np.meshgrid(x, y)

        pr = pyproj.Proj(projection)
        
        lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)
        lon = lon.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        lat = lat.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        
        # Build the xarray dataset
        ds = xr.Dataset(
            data_vars={
                short_name: (['x', 'y'], file_content.get("dataset1/data1/data", np.nan),
                            {'long_name': long_name, 'units': units})
            },
            coords={
                'x': (['x'], x, {'axis': 'X', 'standard_name': 'projection_x_coordinate',
                                'long_name': 'x-coordinate in Cartesian system', 'units': 'm'}),
                'y': (['y'], y, {'axis': 'Y', 'standard_name': 'projection_y_coordinate',
                                'long_name': 'y-coordinate in Cartesian system', 'units': 'm'}),
                'lon': (['y', 'x'], lon, {'standard_name': 'longitude', 'long_name': 'longitude coordinate',
                                        'units': 'degrees_east'}),
                'lat': (['y', 'x'], lat, {'standard_name': 'latitude', 'long_name': 'latitude coordinate',
                                        'units': 'degrees_north'})
            }
        )
        ds['time'] = time

        # Append the dataset to the list
        dataset.append(ds)
        
    # Concatenate datasets along the time dimension
    final_dataset = xr.concat(dataset, dim='time')

    return final_dataset

def prep(field: xr.DataArray) -> np.ndarray:
    '''
    - Crop xarray data to required dimensions (700x700 to 256x256)
    - Reshape it to:
        [B, T, C, H, W] - Batch, Time, Channel, Heigh, Width
    args:
        - field: xarray.DataArray
            The precipitation data variable from the xarray
    '''
    # Crop the center of the field and get a 256x256 image
    # Intervals of +/- 256/2 around the center (which is 700/2)
    field=field['precip_intensity']
    low = (700//2) - (256//2)
    high = (700//2) + (256//2)
    cropped = field[:, low:high, low:high]
    
    # Passing a tuple to expand_dims leads to
    # two dimensions of 1 added at those indeces in the array.shape (Batch and Channel)
    expanded = np.expand_dims(cropped.to_numpy(), (0,2))
    
    #reshape it so that it should have a size (22,256,256,1), where 22 stands for the 4 inputs and 18 next observations
    return expanded.reshape(22,256,256,1)






def forecast_demo():
    # get the cache directory where the data will be stored
    if os.name == "nt":  # Window
        cache_dir = os.path.join(os.path.expanduser("~"), "pysteps", "data")
    else:  # Unix
        cache_dir = os.path.join(os.path.expanduser("~"), ".pysteps", "data")

    if not os.path.exists(cache_dir):
        print("Downloading test_data")
        # Download the entire repository to the cache directory
        os.makedirs(cache_dir, exist_ok=True)
        print(cache_dir)
        pysteps.datasets.download_pysteps_data(cache_dir,force=True)
        cache_dir=os.path.join(cache_dir,"/radar/rmi/radqpe/20210704")
        print(f"saved in {cache_dir}")
    else:
        cache_dir=r"C:\Users\user\pysteps\data\radar\rmi\radqpe\20210704"
        


    
    

    
    data=prep(read(cache_dir))
    # slice the first four frames for predictions
    input_frame=data[:4]
    print(input_frame.shape)

    # slice the next 18 frames and reshaping it  for the visualising the observation Vs Forecast 
    observation=data[4:].reshape((18,256,256))
    print(observation.shape)
    # transforming it into a numpy and resizing the forecast to have a shape of (18,256,256) for visualisation
    dgmr_forecast=forecast(input_frame,num_samples=1).numpy().reshape((18,256,256))
    
    print(dgmr_forecast.shape)
    
    animations.animate(observation,dgmr_forecast,savefig=True)



    """
    Inspired by the LDCast, performs a demo on how the DGMR plugin should work, with the expected input and output.
    """


if __name__ == '__main__':
    forecast_demo()




