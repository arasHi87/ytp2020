import sys
import numpy as np
import matplotlib.pyplot as plt

from func import loader, utils
from func.logger import logger

PATH = './data'

if __name__ == '__main__':
    # load data
    datas = loader.load_scans(PATH)
    first_patient = datas['CT1']

    # base data processing
    patient_pixels = loader.get_pixels_hu(first_patient)
    pix_resampled, spacing = loader.resample(patient_pixels, first_patient)

    # start main data processing
    logger.info('Starting processe lcc and fill it')

    for i in range(50, 51):
        logger.info('Now on {}'.format(i))

        lcc = utils.GetConnectComponet(pix_resampled[i])
        pix_resampled[i] = utils.FillConnectComponet(pix_resampled[i], lcc)

    # show data
    loader.plot_3d(pix_resampled, threshold=0)
