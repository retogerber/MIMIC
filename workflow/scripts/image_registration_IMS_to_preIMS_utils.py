import numpy as np
import skimage
from segment_anything import SamPredictor
from wsireg.utils.im_utils import grayscale
import SimpleITK as sitk
import tifffile
import zarr

def normalize_image(image: np.ndarray):
    '''scale image by 0 to 1'''
    return (image-np.nanmin(image))/(np.nanmax(image)- np.nanmin(image))


def readimage_crop(image: str, bbox: list[int]):
    '''Read crop of an image'''
    bbox = [int(b) for b in bbox]
    store = tifffile.imread(image, aszarr=True)
    z = zarr.open(store, mode='r')
    if isinstance(z, zarr.hierarchy.Group): 
        image_crop = z[0][bbox[0]:bbox[2],bbox[1]:bbox[3],:]
    elif isinstance(z, zarr.core.Array): 
        image_crop = z[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    return image_crop



def prepare_image_for_sam(image: np.ndarray, scale): 
    '''Convert image to grayscale, equalize and scale'''
    img = grayscale(image, True)
    img = skimage.transform.rescale(img, scale, preserve_range = True)   
    img = (img-np.nanmin(img))/(np.nanmax(img)- np.nanmin(img))
    img = skimage.exposure.equalize_adapthist(img)
    img = normalize_image(img)*255
    img = img.astype(np.uint8)
    return img

def apply_filter(image: np.ndarray):
    '''Apply sobel filter and scaling'''
    img = skimage.filters.sobel(image)
    img = normalize_image(img)*255
    img = img.astype(np.uint8)
    img = np.stack([img, img, img], axis=2)
    return img

def preprocess_mask(mask: np.ndarray, image_resolution, opening_size=5):
    '''Process image mask and return largest connected region'''
    mask1tmp = skimage.morphology.isotropic_opening(mask, np.ceil(image_resolution*5))
    mask1tmp, count = skimage.measure.label(mask1tmp, connectivity=2, return_num=True)
    counts = np.unique(mask1tmp, return_counts = True)
    countsred = counts[1][counts[0] > 0]
    indsred = counts[0][counts[0] > 0]
    maxind = indsred[countsred == np.max(countsred)][0]
    mask1tmp = mask1tmp == maxind
    return mask1tmp



def sam_core(img: np.ndarray, sam):
    '''Run segment anything model on image'''
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    input_points = np.array([
        [img.shape[0]//2,img.shape[1]//2]
        ])
    input_labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    return masks, scores

def get_max_dice_score(masks1: np.ndarray, masks2: np.ndarray):
    '''For a set of sam output masks get the maximum overlap'''
    dice_scores = np.zeros((masks1.shape[0],masks2.shape[1]))
    for i in range(masks1.shape[0]):
        for j in range(masks2.shape[0]):
            mask_image1 = masks1[i,:,:]
            mask_image2 = masks2[j,:,:]

            union = np.logical_and(mask_image1,mask_image2)
            dice_score = (2*np.sum(union))/(np.sum(mask_image1)+np.sum(mask_image2))
            dice_scores[i,j] = dice_score

    indices = np.where(dice_scores == dice_scores.max())
    h, w = masks1.shape[-2:]
    mask_image1 = masks1[indices[0][0]].reshape(h,w,1)
    h, w = masks2.shape[-2:]
    mask_image2 = masks2[indices[1][0]].reshape(h,w,1)

    return mask_image1, mask_image2, dice_scores[indices[0][0],indices[1][0]]


def smooth_mask(mask: np.ndarray, disklen: int):
    '''Apply mean filter'''
    tmmask = np.zeros((mask.shape[0]+2*disklen+10,mask.shape[1]+2*disklen+10,1))
    tmmask[disklen:(mask.shape[0]+disklen),disklen:(mask.shape[1]+disklen),:] = mask
    tmmaskm = skimage.filters.rank.mean(tmmask[:,:,0].astype(np.uint8)*255, footprint=skimage.morphology.disk(disklen))
    maskm = np.stack([tmmaskm[disklen:(mask.shape[0]+disklen),disklen:(mask.shape[1]+disklen)]],axis=2)
    maskm = maskm.astype(np.double)
    maskm[np.logical_not(mask[:,:,0])] = np.nan
    maskm = normalize_image(maskm)*255
    maskm = maskm.astype(np.uint8)
    return maskm

def dist_centroids(cent1, cent2, rescale):
    '''Calculate euclidean distance between centroids'''
    euclid_dist_pixel = ((cent1[0]-cent2[0])**2 + (cent1[1]-cent2[1])**2)**0.5
    euclid_dist = euclid_dist_pixel*rescale
    return euclid_dist



def create_ring_mask(img, outscale, inscale):
    '''
    return a ring mask of an input mask
    img: boolean image
    outscale: pixels to scale outwards from current border
    inscale: pixels to scale inwards from current border
    '''
    outermask = skimage.morphology.isotropic_dilation(img, outscale)
    tmmask = np.zeros((img.shape[0]+2,img.shape[1]+2))
    tmmask[1:(img.shape[0]+1),1:(img.shape[1]+1)] = img 
    innermask = skimage.morphology.isotropic_erosion(tmmask, inscale)
    innermask = innermask[1:(img.shape[0]+1),1:(img.shape[1]+1)]
    ringmask = np.logical_and(outermask, np.logical_not(innermask))
    return ringmask


# copy paste from: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb
def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    Args:
        composite_transform (SimpleITK.CompositeTransform): Input composite transform which contains only
                                                            global transformations, possibly nested.
        result_center (tuple,list): The desired center parameter for the resulting affine transformation.
                                    If None, then set to [0,...]. This can be any arbitrary value, as it is
                                    possible to change the transform center without changing the transformation
                                    effect.
    Returns:
        SimpleITK.AffineTransform: Affine transformation that has the same effect as the input composite_transform.
    """
    # Flatten the copy of the composite transform, so no nested composites.
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        # The TranslationTransform interface is different from other
        # global transformations.
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            # Some global transformations do not have a translation
            # (e.g. ScaleTransform, VersorTransform)
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c

    return sitk.AffineTransform(A.flatten(), t, c)

