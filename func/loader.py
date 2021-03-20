import os

import numpy as np
import pandas as pd
import scipy.ndimage
import pydicom as dicom

from glob import glob
from .logger import logger
from typing import List, Union
from plotly.offline import plot
from skimage import measure, morphology
from plotly.tools import FigureFactory as FF
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_scans(path: str) -> dict:
    samples: Dict[str, List[list]] = {}
    
    for sample in glob(os.path.join(path, '*')):
        index: str = os.path.basename(sample)
        logger.info(f"starting load {index}")

        samples[index] = [dicom.read_file(x) for x in glob(os.path.join(sample, '*'))]
        samples[index].sort(key=lambda x: int(x.InstanceNumber))
        
        try:
            slice_thickness = np.abs(samples[index][0].ImagePositionPatient[2] - samples[index][1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(samples[index][0].SliceLocation - samples[index][1].SliceLocation)

        for s in samples[index]:
            s.SliceThickness - slice_thickness
        
        logger.info(f"finish load {index}")
        return samples

def get_pixels_hu(slices: list) -> np.ndarray:
    image:np.ndarray = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    logger.info('start transform unit to HU')

    # set out border elements to 0
    image[image == -2000] = 0

    # transform unit to HU
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image: np.int16, scan:List[list], new_spacing:List[int] = [1, 1, 1]) -> Union[np.ndarray, np.ndarray]:
    spacing:np.ndarray = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=400, step_size=1):
    p = image.transpose(2,1,0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    
    x, y, z = zip(*verts)
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)'] 
    fig = FF.create_trisurf(x=x,
			y=y, 
			z=z,
			plot_edges=False,
			show_colorbar=False,
			colormap=colormap,
			simplices=faces,
			backgroundcolor='rgb(64, 64, 64)',
			title="Interactive Visualization")
    
    plot(fig)

