import cv2
import tifffile
import zarr
import numpy as np
import wsireg
import os
import skimage
from sklearn.neighbors import KDTree
import rembg
import segment_anything


def get_pyr_levels(image: str) -> list:
    store = tifffile.imread(image, aszarr=True)
    z = zarr.open(store, mode='r')
    if isinstance(z, zarr.hierarchy.Group): 
        zd = z.attrs['multiscales'][0]['datasets']
        pyr_levels = [int(zde['path']) for zde in zd]
    elif isinstance(z, zarr.core.Array): 
        pyr_levels = [0]
    return pyr_levels

def check_pyr_level(z, pyr_level):
    if 'multiscales' in z.attrs:
        zd = z.attrs['multiscales'][0]['datasets']
        matched_lvl = [str(lv['path'])==str(pyr_level) for lv in zd]
        if any(matched_lvl):
            pyrl = [i for i, x in enumerate(matched_lvl) if x][0]
        else:
            Exception("No multiscale group found in the zarr file.")
    else:
        Exception("No multiscale group found in the zarr file.")
    return pyrl


def readimage_crop(image: str, bbox: list[int], pyr_level=0):
    '''
    Read a cropped portion of an image.

    This function reads an image from a file and returns a cropped portion of it.

    Parameters:
    image (str): The path to the image file to read.
    bbox (list[int]): A list of four integers specifying the bounding box of the crop.
                      The list should be in the format [x1, y1, x2, y2], where (x1, y1) is the
                      bottom-left corner of the bounding box and (x2, y2) is the top-right corner.

    Returns:
    image_crop: The cropped portion of the image.
    '''
    bbox = [int(b) for b in bbox]
    store = tifffile.imread(image, aszarr=True)
    z = zarr.open(store, mode='r')
    if isinstance(z, zarr.hierarchy.Group): 
        pyrl = check_pyr_level(z, pyr_level)
        if z[pyrl].ndim == 3:
            image_crop = z[pyrl][bbox[0]:bbox[2],bbox[1]:bbox[3],:]
        else:
            image_crop = z[pyrl][bbox[0]:bbox[2],bbox[1]:bbox[3]]
    elif isinstance(z, zarr.core.Array): 
        image_crop = z[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    return image_crop


def get_image_shape(image: str, pyr_level=0):
    '''
    Get the shape of an image.

    This function reads an image from a file and returns its shape as a tuple.

    Parameters:
    image (str): The path to the image file to read.

    Returns:
    image_shape (tuple): The shape of the image.
    '''
    store = tifffile.imread(image, aszarr=True)
    z = zarr.open(store, mode='r')
    if isinstance(z, zarr.hierarchy.Group): 
        pyrl = check_pyr_level(z, pyr_level)
        image_shape = z[pyrl].shape
    elif isinstance(z, zarr.core.Array): 
        image_shape = z.shape
    return image_shape


def saveimage_tile(image: np.ndarray, filename: str, resolution: float, dtype= np.uint8, is_rgb: bool = None):
    '''
    Save an image as a tiled OME-TIFF file.

    This function takes an image, its filename, and resolution as input, and saves the image as a tiled OME-TIFF file.

    Parameters:
    image (np.ndarray): The image to be saved. It should be a 2D numpy array.
    filename (str): The path to the output file. The basename of this path is used as the base name for the OME-TIFF planes.
    resolution (float): The resolution of the image in pixelsize in microns.

    Returns:
    None. The function writes the image to disk and does not return any value.
    '''
    empty_transform = wsireg.parameter_maps.transformations.BASE_RIG_TFORM
    empty_transform['Spacing'] = (str(resolution),str(resolution))
    if not is_rgb is None and not is_rgb:
        empty_transform['Size'] = (image.shape[2], image.shape[1])
    else:
        empty_transform['Size'] = (image.shape[1], image.shape[0])
    rt = wsireg.reg_transforms.reg_transform_seq.RegTransform(empty_transform)
    rts = wsireg.reg_transforms.reg_transform_seq.RegTransformSeq(rt,[0])
    ri = wsireg.reg_images.loader.reg_image_loader(image.astype(dtype), resolution)
    if not is_rgb is None:
        ri._is_rgb = is_rgb
    writer = wsireg.writers.ome_tiff_writer.OmeTiffWriter(ri, reg_transform_seq=rts)
    img_basename = os.path.basename(filename).split(".")[0]
    img_dirname = os.path.dirname(filename)
    writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)


def convert_and_scale_image(image: np.ndarray, scale: float=1.0) -> np.ndarray: 
    '''
    Convert an image to grayscale, equalize its histogram, and scale it.

    Parameters:
    image (np.ndarray): The input image. It should be a 3D numpy array.
    scale (float): The scale factor to resize the image. If it's 1, the image is not resized.

    Returns:
    img (np.ndarray): The processed image. It's a grayscale image with enhanced contrast and possibly resized.
    '''
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    if scale != 1:
        wn = int(img.shape[0]*scale)
        hn = int(img.shape[1]*scale)
        img = cv2.resize(img, (hn,wn), interpolation=cv2.INTER_NEAREST)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.createCLAHE().apply(img)
    return img


def subtract_postIMS_grid(img: np.ndarray) -> np.ndarray:
    # adapted from: https://stackoverflow.com/a/63403618
    # read input as grayscale
    # opencv fft only works on grayscale
    hh, ww = img.shape[:2]

    # convert image to floats and do dft saving as complex output
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

    # apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft)

    # extract magnitude and phase images
    mag, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

    # get spectrum for viewing only
    spec = np.log(mag)

    # get local threshold
    local_thresh = skimage.filters.threshold_local(spec, block_size=191)
    threshold = np.quantile((spec-local_thresh).flatten(),0.99)
    # threshold based on difference from local threshold to image
    mask = (spec-local_thresh) > threshold

    # remove individual high pixels
    mask = cv2.medianBlur(mask.astype(np.uint8)*255, 3)
    # increase peaks
    mask = cv2.morphologyEx(src=mask.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.diamond(1)).astype(bool).astype(np.uint8)

    # detect regions and get distance of centroids to calculate radius
    _, _, stats, cents = cv2.connectedComponentsWithStatsWithAlgorithm(mask.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S, ccltype=cv2.CCL_SAUF)
    kdt = KDTree(cents, leaf_size=30, metric='euclidean')
    dists, indices = kdt.query(cents, k=2, return_distance=True)
    radius = int(np.median(dists[:,1])*0.25)

    # blacken out center DC region from mask
    cx = ww // 2
    cy = hh // 2
    mask = cv2.circle(mask, (cx,cy), radius, 0, -1)

    # invert mask
    mask = 1 - mask
    mask=mask.astype(np.uint8)

    # apply mask to magnitude image
    mag_notch = mask*mag
    
    ksize = int(np.median(dists[:,1])*0.25)
    # get local average magnitude
    tmp_mag_notch = mag_notch.copy()
    tmp_mag_notch[mask==0] = 0
    tmp_mag_notch = cv2.circle(tmp_mag_notch, (cx,cy), radius, 0, -1)
    magmedian = cv2.medianBlur(tmp_mag_notch, ksize=5)
    magmedian = cv2.blur(magmedian, ksize=(ksize,ksize))
    magmedian = cv2.medianBlur(magmedian, ksize=5)

    # replace masked regions with local average
    # mag_notch[mask==0] = local_thresh[mask==0]*magmedian[mask==0]
    mag_notch[mask==0] = magmedian[mask==0]

    # convert magnitude and phase into cartesian real and imaginary components
    real, imag = cv2.polarToCart(mag_notch, phase)

    # combine cartesian components into one complex image
    complex = cv2.merge([real, imag])

    # shift origin from center to upper left corner
    complex_ishift = np.fft.ifftshift(complex)

    # do idft with normalization saving as real output
    img_notch = cv2.idft(complex_ishift, flags=cv2.DFT_SCALE+cv2.DFT_REAL_OUTPUT)

    # equalize histogram
    img_notch = cv2.normalize(src=img_notch, dst=img_notch, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_notch = cv2.createCLAHE().apply(img_notch)

    return img_notch


def extract_mask(file: str, bb: list, session = None, rescale: int = 1, is_postIMS: bool = False, sam=None, pts: np.ndarray = None, multiple_rembgth: bool = False, pyr_level=0) -> np.ndarray:
    '''
    Extract a tissue mask from an image file.

    This function takes an image file, a bounding box, an optional rembg session, a rescale factor, and a boolean indicating
    whether the image is post-IMS as input. It reads and crops the image based on the bounding box, rescales the image,
    and if the image is postIMS it subtracts the IMS grid.

    The function then removes the background using rembg to create a mask. It removes small holes
    in the mask and applies a morphological closing operation to the mask.

    Parameters:
    file (str): The path to the image file.
    bb (list): The bounding box to apply when cropping the image.
    session: An optional session for the background removal operation.
    rescale (int): The factor by which to rescale the image.
    is_postIMS (bool): Whether the image is post-IMS.

    Returns:
    masks (np.ndarray): The extracted mask. It's a 3D binary array.
    '''
    if session == None and sam == None:
        session = rembg.new_session("isnet-general-use")
    w = readimage_crop(file, bb, pyr_level=pyr_level)
    w = convert_and_scale_image(w, rescale)
    if is_postIMS:
        w = subtract_postIMS_grid(w)
        w = cv2.blur(w, (9,9))
    w = cv2.fastNlMeansDenoising(w, None, 10, 7, 21)
    w = np.stack([w, w, w], axis=2)
    if sam == None:
        wr = rembg.remove(w, only_mask=True, session=session)
        try:
            th = skimage.filters.threshold_minimum(wr, nbins=256)
        except:
            th = skimage.filters.threshold_otsu(wr, nbins=256)
        if multiple_rembgth:
            ths = [np.quantile(wr[wr>0].flatten(),q) for q in [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]]
            ths = [th] + ths
        else:
            ths = [th]

        masks = []
        for th in ths:
            tmasks = wr>th
            tmasks = skimage.morphology.remove_small_holes(tmasks,100**2*np.pi)
            tmasks = cv2.morphologyEx(tmasks.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)).astype(bool)
            tmasks = np.stack([tmasks])
            masks.extend(tmasks)
        masks = np.array(masks)
    else:
        masks, scores = sam_core(w, sam, pts=pts)
        masks = np.stack([preprocess_mask(msk,1) for msk in masks ])>0
    return masks


def sam_core(img: np.ndarray, sam, pts: np.ndarray = None) -> (np.ndarray, np.ndarray):
    '''
    Run the Segment Anything Model (SAM) on an image.

    As input to SAM a single point is used, which is the center of the image.

    Parameters:
    img (np.ndarray): The input image. It should be a 3D numpy array.
    sam: The SAM model to use for segmentation.

    Returns:
    masks (np.ndarray): The segmentation masks produced by the SAM predictor. Each mask is a binary image that represents a different segment of the input image.
    scores (np.ndarray): The scores assigned by the SAM predictor to each mask. Higher scores indicate more confident predictions.
    '''
    predictor = segment_anything.SamPredictor(sam)
    predictor.set_image(img)

    if pts is None:
        input_points = np.array([
            [img.shape[0]//2,img.shape[1]//2]
            ])
        input_labels = np.array([1])
    else:
        input_points = pts
        input_labels = np.ones(pts.shape[0])
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    return masks, scores 


def preprocess_mask(mask: np.ndarray, image_resolution: float = 1.0) -> np.ndarray:
    '''
    Preprocess an image mask and return the largest connected region.

    Parameters:
    mask (np.ndarray): The input image mask. It should be a 2D numpy array.
    image_resolution (float): The resolution of the image. This is used to determine the size of the structuring element.

    Returns:
    mask1tmp (np.ndarray): The processed mask. It's a binary image that only includes the largest connected component.
    '''
    kernel_size = int(np.ceil(image_resolution*5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask1tmp = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    _, labels = cv2.connectedComponents(mask1tmp)
    counts = np.bincount(labels.flatten())
    if len(counts)>1:
        maxind = np.argmax(counts[1:]) + 1  # Ignore background
        mask1tmp = (labels == maxind).astype(np.uint8)
    return mask1tmp

