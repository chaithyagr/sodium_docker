"""
Script to carry out reconstruction given a set of parameters
Author: Chaithya G R
"""
# Package import
import utils
from mri.reconstructors import SingleChannelReconstructor

from mri.operators import NonCartesianFFT, WaveletN
from mri.operators.fourier.utils import estimate_density_compensation
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps

from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD, convert_NCNSxD_to_NCxNSxD, \
    interpolate_shots
# Third party import
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
from modopt.math.metrics import ssim

# Third party import
import os
import numpy as np
import datetime
import pickle as pkl
import nibabel as nib

def roll_ssim(ref, test, mask, roll=50):
    return ssim(ref, np.roll(test, roll), mask)

# Loading input data
if __name__ == "__main__":
    kspace_loc, kspace_data, out_dir, file, params = utils.load_input_from_args(
        load_params=True,
        params_file_kwargs={'file': 'selfcalibrating_recon.json'}
    )
 #   params['vol_shape'] = (N, N, Nz)

 
    regularizer_op = SparseThreshold(Identity(), 0, thresh_type="soft")

    linear_op = utils.str_to_class(params['linear_params']['init_class'])(
        **params['linear_params']['kwargs']
    )
    density_comp = estimate_density_compensation(kspace_loc, params['vol_shape'])
    # Smaps, SOS = get_Smaps(
    #     k_space=kspace_data,
    #     img_shape=params['vol_shape'],
    #     samples=kspace_loc,
    #     min_samples=kspace_loc.min(axis=0),
    #     max_samples=kspace_loc.max(axis=0),
    #     density_comp=density_comp,
    #     **params['smaps_params']['kwargs'],
    # )
    # Setup Fourier Operator with SENSE
    # fourier_op_sense = utils.str_to_class(params['fourier_params']['init_class'])(
    #     samples=kspace_loc,
    #     shape=params['vol_shape'],
    #     n_coils=kspace_data.shape[0],
    #     smaps=Smaps,
    #     density_comp=density_comp,
    #     **params['fourier_params']['kwargs'],
    # )
    fourier_op = NonCartesianFFT(
        samples=kspace_loc,
        shape=params['vol_shape'],
        implementation='gpuNUFFT',
        density_comp=density_comp,
    )
   
    reconstructor = SingleChannelReconstructor(
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=regularizer_op,
        num_check_lips=0,
        verbose=1,
    )
    reconstructor.prox_op.weights = 0.000000008

    x_final = reconstructor.reconstruct(
        kspace_data=kspace_data,
        **params['optimizer_params']['kwargs'],

      #  metrics=metrics,
      #  metric_call_period=1,
    )

    currentDT = datetime.datetime.now()
    filename = os.path.join(
        out_dir,
        file +
        "_D" + str(currentDT.day) + "M" + str(currentDT.month) +
        "Y" + str(currentDT.year) + "T" + str(currentDT.hour)
    ) + ".pkl"
    pkl.dump((x_final, params, reconstructor.gradient_op.spec_rad),
             open(filename, 'wb'), protocol=4)

    img = nib.Nifti1Image(np.abs(x_final[0]) , np.eye(4))
    filename = os.path.join(
        out_dir,
        file +
        "_D" + str(currentDT.day) + "M" + str(currentDT.month) +
        "Y" + str(currentDT.year) + "T" + str(currentDT.hour)
    ) +str(reconstructor.prox_op.weights) + ".nii"
    nib.save(img, filename)