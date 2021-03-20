import os
import sys
import pydicom
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from glob import glob
from logger import logger 
from skimage import measure
from plotly.offline import plot
from plotly.tools import FigureFactory as FF
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class FaceDestroy():
    def __init__(self, path, amount=1):
        self.samples = dict()

        for d_path in glob(os.path.join(path, '*')):
            name = d_path.split('/')[-1]

            logger.info('Loading slices......')

            self.samples[name] = list()
            self.samples[name] = [pydicom.read_file(x) for x in glob(os.path.join(d_path, '*'))]
            
            logger.info('Load finished')
            self.samples[name].sort(key=lambda x: int(x.InstanceNumber))

            try:
                slice_thickness = np.abs(self.samples[name][0].ImagePositionPatient[2] - self.samples[name][1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(self.samples[name][0].SliceLocation - self.samples[name][1].SliceLocation)
           
            for s in self.samples[name]:
                s.SliceThickness = slice_thickness

    def get_pixels_hu(self):
        result = dict()

        for key, scans in self.samples.items():
            image = np.stack([s.pixel_array for s in scans])
            image = image.astype(np.int16)

            # Set outside-of-scan pixels to 1
            # The intercept is usually -1024, so air is approximately 0
            image[image == -2000] = 0

            # Convert to Hounsfield units (HU)
            intercept = scans[0].RescaleIntercept
            slope = scans[0].RescaleSlope

            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(np.int16)

            image += np.int16(intercept)
            result[key] = np.array(image, dtype=np.int16)

        return result

    def show_voxel_histogram(self):
        for key, val in self.get_pixels_hu().items():
            imgs_to_process = val.astype(np.float64) 
            
            plt.hist(imgs_to_process.flatten(), bins=50, color='c')
            plt.xlabel("Hounsfield Units (HU)")
            plt.ylabel("Frequency")
            plt.show()

    def show_image_stack(self, rows=6, cols=6, start_with=10, show_every=5):
        for key, stack in self.get_pixels_hu().items():
            fig, ax = plt.subplots(rows, cols, figsize=[12,12])
            
            for i in range(rows*cols):
                ind = start_with + i*show_every
                ax[int(i/rows), int(i % rows)].set_title('slice %d' % ind)
                ax[int(i/rows), int(i % rows)].imshow(stack[ind], cmap='gray')
                ax[int(i/rows), int(i % rows)].axis('off')

            plt.show()

    def resample(self, new_spacing=[1, 1, 1]):
        images = self.get_pixels_hu()
        result = dict()

        for key, image in images.items():
            logger.info('Starting resample {}'.format(key))
            
            scan = self.samples[key]

            # Determine current pixel spacing
            spacing = map(float, list([scan[0].SliceThickness] + list(scan[0].PixelSpacing))) 
            spacing = np.array(list(spacing))

            resize_factor = spacing / new_spacing
            new_real_shape = image.shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / image.shape
            new_spacing = spacing / real_resize_factor
            
            image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
            result[key] = image

            logger.info('Finish resample {}'.format(key))
        
        return result

    def make_mesh(self, threshold=400, step_size=1):
        images = self.resample()
        result = dict()

        for key, image in images.items():
            p = image.transpose(2, 1, 0)
            verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
            result[key] = (verts, faces)

        return result
    
    def plotly_3d(self):
        meshes = self.make_mesh()

        for key, val in meshes.items():
            logger.info('Drawing {}'.format(key))
            
            verts, faces = val
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


if __name__ == '__main__':
    FD = FaceDestroy('./data')
    #  FD.show_image_stack()
    #  FD.plotly_3d()
    
    for key, stack in FD.get_pixels_hu().items():
        print(stack)
