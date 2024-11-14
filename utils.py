"""Script to just read data and kspace_loc"""

import matplotlib
matplotlib.use('Agg')
import argparse
import warnings
import os, json, sys
import glob
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pydicom
from scipy.interpolate import interpn

def plot_view(image):
    plt.imshow(np.abs(image), cmap='gray')
    plt.axis('off')
    plt.grid(b=None)


def plot_image(image, pos_x, pos_y, num_pics_x, num_pics_y, f, ssim):
    pos_x = pos_x / num_pics_x
    num_pics_x = 2*num_pics_x
    pos_y = pos_y /num_pics_y
    ax = f.add_axes([pos_x + 0, 1-1/num_pics_y - pos_y, 1/num_pics_x, 1/num_pics_y])
    ax.imshow(np.flipud(image[60:-60, 50:-10, image.shape[2]//2].T), cmap='gray')
    plt.title("SSIM = " + str(np.around(ssim, 4)), fontsize=16)
    ax.set_axis_off()
    ax = f.add_axes([pos_x + 1/num_pics_x, 1-1/num_pics_y/2 - pos_y, 1/num_pics_x, 1/num_pics_y/2])
    ax.imshow(image[60:-55, image.shape[1]//2, :].T, cmap='gray')
    ax.set_axis_off()
    ax = f.add_axes([pos_x + 1/num_pics_x, 1-1/num_pics_y - pos_y, 1/num_pics_x, 1/num_pics_y/2])
    ax.imshow(np.fliplr(image[image.shape[0]//2, 50:-10, :].T), cmap='gray')
    ax.set_axis_off()


def view_compare(views, titles, split=4, black_back=True):
    if black_back:
        plt.style.use('dark_background')
    vol_shape = views[0].shape
    num_views = min(len(views), split)
    for i, (view, title) in enumerate(zip(views, titles)):
        if np.mod(i, split) == 0:
            fig = plt.figure(figsize=(17, 15))
        ctr = np.mod(i, split)
        ax = plt.subplot(2, num_views, ctr+1)
        plot_view(np.flipud(view[:, :, vol_shape[2] // 2].T))
        ax.set_title(title)
        plt.subplot(4, num_views, num_views*2 + ctr+1)
        plot_view(view[:, vol_shape[1]//2, :].T)
        plt.subplot(4, num_views, num_views*3 + ctr+1)
        plot_view(view[vol_shape[0]//2, :, :].T)


def get_raw_data(filename):
    # Function that reads a SIEMENS .dat file and returns a k space data
    from twixreader import Twix
    data = Twix(filename)[-1]['ima'].raw()
    data = np.moveaxis(data, 1, 0)
    data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
    return data



def add_phase_kspace(kspace_data, kspace_loc, shifts={}):
    if shifts == {}:
        shifts = (0,) * kspace_loc.shape[1]
    if len(shifts) != kspace_loc.shape[1]:
        raise ValueError("Dimension mismatch between shift and kspace locations! "
                         "Ensure that shifts are right")
    phi = np.zeros_like(kspace_data)
    for i in range(kspace_loc.shape[1]):
        phi += kspace_loc[:, i] * shifts[i]
    phase = np.exp(-2 * np.pi * 1j * phi)
    return kspace_data * phase


def get_samples(filename, num_adc_samples=4096, dwell_time=0.005, gamma=42.583, Kmax=None, verbose=1):
    from mri.operators.utils import normalize_frequency_locations
    from sparkling.utils.gradient import get_kspace_loc_from_gradfile
    sample_locations, params = get_kspace_loc_from_gradfile(
        filename,
        dwell_time=dwell_time,
        num_adc_samples=num_adc_samples,
        gamma=gamma,
    )
    num_shots = sample_locations.shape[0]
    num_samples_per_shot = sample_locations.shape[1]
    dimension = sample_locations.shape[2]
    sample_locations = np.reshape(sample_locations, (num_samples_per_shot * num_shots, dimension))
    if verbose:
        print(np.max(sample_locations, axis=0))
        print(np.min(sample_locations, axis=0))
    return normalize_frequency_locations(sample_locations, Kmax=Kmax)


def load_json_params(file=None, islocal=True):
    if file is None:
        file = 'selfcalibrating_recon.json'
    if islocal:
        file = os.path.join(os.path.dirname(__file__), 'configs', file)
    with open(file, 'r') as fp:
        params = json.load(fp)
    return params


def str_to_class(classname):
    return getattr(sys.modules['__main__'], classname)


def get_hanning_filter(vol_shape):
    """ create hanning filter

    Parameters
    ----------
    volume_shape: np.ndarray
        size of the resulting hanning filter.

    Returns
    -------
    hanning: np.ndarray
        the hanning filter
    """

    dim = len(vol_shape)
    # create hanning filter
    hanning = np.hanning(vol_shape[0])
    for i in range(1, dim):
        hanning = np.expand_dims(hanning, -1)
        hanning = np.dot(hanning, np.expand_dims(np.hanning(
            vol_shape[i]), -1).T)
    return hanning


def virtual_coil_reconstruction(imgs):
    """
    Calculate the combination of all coils using virtual coil reconstruction

    Parameters
    ----------
    imgs: np.ndarray
        The images reconstructed channel by channel
        in shape [Nch, Nx, Ny, Nz] for 3D or [Nch, Nx, Ny] for 2D

    Returns
    -------
    img_comb: np.ndarray
        The combination of all the channels in a complex valued
        in shape [Nx, Ny, Nz] for 3D or [Nx, Ny] for 2D
    """
    img_sh = imgs.shape
    # Compute first the virtual coi
    weights = np.sum(np.abs(imgs), axis=0)
    weights[weights == 0] = 1e-16
    phase_reference = np.asarray([np.angle(np.sum(
        imgs[ch].flatten())) for ch in range(img_sh[0])])
    reference = np.asarray([(imgs[ch] / weights) / np.exp(
        1j * phase_reference[ch]) for ch in range(32)])
    virtual_coil = np.sum(reference, axis=0)
    difference_original_vs_virtual = np.conjugate(imgs) * virtual_coil
    # Hanning filtering in readout and phase direction
    hanning = get_hanning_filter([img_sh[1], img_sh[2]])
    if len(img_sh) > 3:
        hanning = np.tile(np.expand_dims(np.tile(
            hanning, (img_sh[0], 1, 1)), -1), (1, 1, 1, img_sh[3]))
    # Removing the background noise via low pass filtering
    difference_original_vs_virtual = np.fft.ifft2(np.fft.fft2(
        difference_original_vs_virtual, axes=(1, 2)) * np.fft.fftshift(
        hanning), axes=(1, 2))
    img_comb = np.asarray([imgs[ch] * np.exp(1j * np.angle(
        difference_original_vs_virtual[ch])) for ch in range(img_sh[0])])
    return np.sum(img_comb, 0)


def get_SWI(imTot):
    """ Process SWI images from complex reconstructed image per channels

        Parameters
        ----------
        image_ch: np.ndarray
            Images reconstructed per channels
            in shape [Nch, Nx, Ny, Nz] for 3D or [Nch, Nx, Ny] for 2D

        Returns
        -------
        SWI: np.ndarray
            SWI image
            in shape [Nx, Ny, Nz] for 3D or [Nx, Ny] for 2D
    """

    # 2. apply hanning filter on readout&phase direction from a kspace portion
    hanning_filt = np.zeros([imTot.shape[0], imTot.shape[1]])
    p_x = [int(np.round(3 * imTot.shape[0] / 10)),
           int(np.round(7 * imTot.shape[0] / 10))]
    p_y = [int(np.round(3 * imTot.shape[1] / 10)),
           int(np.round(7 * imTot.shape[1] / 10))]
    hanning_filt[p_x[0]:p_x[1], p_y[0]:p_y[1]] = get_hanning_filter(
        [p_x[1]-p_x[0], p_y[1]-p_y[0]])
    if len(imTot.shape) == 3:
        hanning_filt = np.tile(np.expand_dims(
                hanning_filt, -1), (1, 1, imTot.shape[2]))
    # get get low-pass filtered phase image
    imTot_filtered = np.fft.ifftn(np.fft.fftn(imTot) * np.fft.fftshift(hanning_filt))
    # compute homodyne image
    imTot_homodyne = imTot * np.exp(-1j * np.angle(imTot_filtered))

    # 3. get phase mask
    imTot_filt_phase = np.angle(imTot_homodyne)
    pha_mask_raw = np.ones(imTot_filt_phase.shape)
    pha_mask_raw[imTot_filt_phase > 0] = (
            1 - imTot_filt_phase[imTot_filt_phase > 0] / np.pi)
    pha_mask_raw_final = pha_mask_raw ** 4
    pos_pha_mask_raw = np.ones(imTot_filt_phase.shape)
    pos_pha_mask_raw[imTot_filt_phase < 0] = (
            1 + imTot_filt_phase[imTot_filt_phase < 0] / np.pi)
    pos_pha_mask_raw_final = pos_pha_mask_raw ** 4

    # 4. compute SWI
    swi = np.abs(imTot) * pha_mask_raw_final #* pos_pha_mask_raw_final

    return swi


def get_mIP(swi, width=20):
    mip = np.zeros_like(swi)
    for i in range(swi.shape[-1]):
        low = max(0, i-width//2)
        high = min(swi.shape[-1], i+width//2)
        mip[:,:,i] = np.min(swi[:, :, low:high], axis=-1)
    return mip


def interpolate_along_z(data, order=2):
    current_xyz = [
        np.linspace(1, 100, i)
        for i in data.shape
    ]
    new_shape = (*data.shape[0:2], data.shape[2]*order)
    new_xyz = [
        np.linspace(1, 100, i)
        for i in new_shape
    ]
    grid = np.meshgrid(*new_xyz, indexing='ij')
    return interpn(tuple(current_xyz), data, tuple(grid))


def load_cartesian_data3D(dcm_path, vol_shape=(384, 384, 208)):
    dcms = os.listdir(dcm_path)
    dcms = sorted(dcms, key=lambda x: int(x.split('.')[4]))
    ref_data = np.zeros(vol_shape)
    for i, dcm in enumerate(dcms):
        dcm_data = pydicom.dcmread(os.path.join(dcm_path,dcm))
        ref_data[:, :, vol_shape[-1]-i-1] = np.flipud(np.fliplr(np.asarray(dcm_data.pixel_array).T))
    return ref_data


def get_traj_file_loc(obs_file):
    try:
        traj_num = int(str.split(os.path.basename(obs_file), '_')[-1][1:-4])
    except:
        traj_num = -1
    candidate_traj = glob.glob(os.path.join(os.path.dirname(obs_file), '..', 'traj', '[0+]' + str(traj_num) + '_*.bin'))
    if len(candidate_traj) > 1:
        warnings.warn('More than one traj file found, using first!')
    elif len(candidate_traj) == 0:
        warnings.warn('Trajectory not found, using from args.loc!')
        return -1
    return candidate_traj[0]


def get_osf(obs_file):
    try:
        osf = int(str.split(str.split(os.path.basename(obs_file), 'OS')[-1], '_')[0])
    except:
        warnings.warn("Could not find OSF, running with default "
                          "dwell_time and Ns provided")
        osf = -1
    return osf


def get_volshape(traj_file, default_volshape):
    try:
        vol_shape = str.split(
            str.split(str.split(os.path.basename(traj_file), 'N')[1], '_')[0],
            'x'
        )
        vol_shape = tuple(int(shp) for shp in vol_shape)
    except:
        warnings.warn("Could not get Matrix size, sticking to default")
        vol_shape = default_volshape
    return vol_shape


def get_fov(traj_file, default_fov):
    try:
        fov = str.split(
            str.split(str.split(os.path.basename(traj_file), 'FOV')[-1], '_')[0],
            'x'
        )
        fov = tuple(int(shp) for shp in fov)
    except:
        warnings.warn("Could not get FOV size, sticking to default")
        vol_shape = default_fov
    return vol_shape


def load_input_from_args(savefile=False, load_params=False,
                         params_file_kwargs={}, verbose=1):
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ns", default=1248)
    parser.add_argument("--dwell_time", default=0.010)
    parser.add_argument("--gamma", default=11.26e3)
    parser.add_argument("--obs",
                        help="Kspace Observations DataFile")
    parser.add_argument("--loc",
                        help="Kspace Location MatFGile")
    parser.add_argument("--img_size", default="80x80x80", help="Matrix Size")
    parser.add_argument("--fov", default="0.24x0.24x0.24", help="FOV")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--n_average", help=None)
    parser.add_argument("--params", default=None, help="parameters file")
    parser.add_argument("--verbose", default='1', help="Verbosity")
    args = parser.parse_args()
    if args.outdir is None:
        raise ValueError("The outdir must point to appropriate ")
    print("Arguments : " + str(args))
    img_size = [int(i) for i in str.split(args.img_size, 'x')]
    fov = [float(i) for i in str.split(args.fov, 'x')]
    num_samples = int(args.Ns)
    dwell_time = float(args.dwell_time)
    gamma = float(args.gamma)
    out_dir = args.outdir
    file_name = args.obs
    n_average = int(args.n_average)
    file = os.path.splitext(os.path.basename(file_name))[0]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(os.path.join(out_dir, file + '_data.pkl')):
        (kspace_loc, kspace_data) = pkl.load(open(os.path.join(out_dir, file + '_data.pkl'), 'rb'))
    else:
        if args.obs is None:
            raise ValueError("The obs must point to right directories of kspace dat")
        mask_name = get_traj_file_loc(args.obs)
        if mask_name == -1:
            if args.loc is None:
                raise ValueError("Could not locate trajectory file!")
                exit(1)
            else:
                mask_name = args.loc
        osf =  5 #  get_osf(mask_name) 
        if osf != -1:
            dwell_time = 0.01/osf
            num_samples = int(args.Ns)*osf
        img_size = get_volshape(mask_name, img_size)
        fov = get_fov(mask_name, fov)
        file_name = args.obs
        file = os.path.splitext(os.path.basename(file_name))[0]
        kspace_data = get_raw_data(file_name)

        kspace_loc = get_samples(
            mask_name,
            num_adc_samples=num_samples,
            dwell_time=dwell_time,
            gamma=gamma,
            Kmax=np.array(img_size) / 2 / np.array(fov),
        )
        if savefile:
            pkl.dump((kspace_loc, kspace_data),
                     open(os.path.join(out_dir, file + '_data.pkl'), 'wb'),
                     protocol=4)
        if load_params:
            params = load_json_params(**params_file_kwargs)
            if 'vol_shape' not in params:
                params['vol_shape'] = img_size
                params['fov'] = fov
            if verbose:
                print(params)
            return kspace_loc, kspace_data, out_dir, file, params, n_average
        else:
            return kspace_loc, kspace_data, out_dir, file, n_average
