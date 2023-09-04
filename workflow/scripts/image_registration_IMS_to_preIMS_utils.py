import numpy as np
from typing import Union
import cv2
import skimage
from segment_anything import SamPredictor
from wsireg.utils.im_utils import grayscale
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform_seq import RegTransform, RegTransformSeq
from wsireg.parameter_maps.transformations import BASE_RIG_TFORM
from wsireg.reg_images.loader import reg_image_loader
import os.path
import SimpleITK as sitk
import tifffile
import zarr
import shapely
from sklearn.neighbors import KDTree

def normalize_image(image: np.ndarray):
    '''scale image by 0 to 1'''
    return (image-np.nanmin(image))/(np.nanmax(image)- np.nanmin(image))


def readimage_crop(image: str, bbox: list[int]):
    '''Read crop of an image'''
    bbox = [int(b) for b in bbox]
    store = tifffile.imread(image, aszarr=True)
    z = zarr.open(store, mode='r')
    if isinstance(z, zarr.hierarchy.Group): 
        if z[0].ndim == 3:
            image_crop = z[0][bbox[0]:bbox[2],bbox[1]:bbox[3],:]
        else:
            image_crop = z[0][bbox[0]:bbox[2],bbox[1]:bbox[3]]
    elif isinstance(z, zarr.core.Array): 
        image_crop = z[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    return image_crop

def get_image_shape(image: str):
    '''Read shape of an image'''
    store = tifffile.imread(image, aszarr=True)
    z = zarr.open(store, mode='r')
    if isinstance(z, zarr.hierarchy.Group): 
        image_shape = z[0].shape
    elif isinstance(z, zarr.core.Array): 
        image_shape = z.shape
    return image_shape


def saveimage_tile(image: np.ndarray, filename: str, resolution: float):
    empty_transform = BASE_RIG_TFORM
    empty_transform['Spacing'] = (str(resolution),str(resolution))
    empty_transform['Size'] = (image.shape[1], image.shape[0])
    rt = RegTransform(empty_transform)
    rts = RegTransformSeq(rt,[0])
    ri = reg_image_loader(image.astype(np.uint8), resolution)
    writer = OmeTiffWriter(ri, reg_transform_seq=rts)
    img_basename = os.path.basename(filename).split(".")[0]
    img_dirname = os.path.dirname(filename)
    writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)



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
    # outermask = skimage.morphology.isotropic_dilation(img, outscale)
    outermask = cv2.morphologyEx(src=img.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int(outscale))).astype(bool)
    tmmask = np.zeros((img.shape[0]+2,img.shape[1]+2))
    tmmask[1:(img.shape[0]+1),1:(img.shape[1]+1)] = img 
    # innermask = skimage.morphology.isotropic_erosion(tmmask, inscale)
    innermask = cv2.morphologyEx(src=tmmask.astype(np.uint8), op = cv2.MORPH_ERODE, kernel = skimage.morphology.square(2*int(inscale))).astype(bool)
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


def create_imz_coords(imzimg: np.ndarray, mask: Union[None, np.ndarray], imzrefcoords: np.ndarray, bbox, rotmat):
    # create coordsmatrices for IMS
    indmatx = np.zeros(imzimg.shape)
    for i in range(imzimg.shape[0]):
        indmatx[i,:] = list(range(imzimg.shape[1]))
    indmatx = indmatx.astype(np.uint32)
    indmaty = np.zeros(imzimg.shape)
    for i in range(imzimg.shape[1]):
        indmaty[:,i] = list(range(imzimg.shape[0]))
    indmaty = indmaty.astype(np.uint32)

    xminimz = bbox[0]
    yminimz = bbox[1]
    xmaximz = bbox[2]
    ymaximz = bbox[3]

    # create coordinates for registration
    if mask is None:
        imzxcoords = indmatx[xminimz:xmaximz,yminimz:ymaximz].flatten()
        imzycoords = indmaty[xminimz:xmaximz,yminimz:ymaximz].flatten()
    else: 
        imzxcoords = indmatx[xminimz:xmaximz,yminimz:ymaximz][mask]
        imzycoords = indmaty[xminimz:xmaximz,yminimz:ymaximz][mask]
    imzcoords = np.stack([imzycoords, imzxcoords],axis=1)

    center_point=np.max(imzrefcoords,axis=0)/2
    imzrefcoords = np.dot(rotmat, (imzrefcoords - center_point).T).T + center_point

    # filter for coordinates that are in data
    in_ref = []
    for i in range(imzcoords.shape[0]):
        in_ref.append(np.any(np.logical_and(imzcoords[i,0] == imzrefcoords[:,0],imzcoords[i,1] == imzrefcoords[:,1])))

    in_ref = np.array(in_ref)
    imzcoords = imzcoords[in_ref,:]
    return imzcoords

def get_rotmat_from_angle(rot):
    # rotate IMS coordinates 
    if rot in [-180, 180]:
        rotmat = np.asarray([[-1, 0], [0, -1]])
    elif rot in [90, -270]:
        rotmat = np.asarray([[0, 1], [-1, 0]])
    elif rot in [-90, 270]:
        rotmat = np.asarray([[0, -1], [1, 0]])
    else:
        rotmat = np.asarray([[1, 0], [0, 1]])
    return rotmat

def get_sigma(threshold:float = 1, 
              p:float = 0.99):
    """Estimate a sigma for gaussian filter, i.e. density (p) at distance (threshold)"""
    return 1/(5.5556*(1-((1-p)/p)**0.1186)/threshold)


def image_from_points(shape, points: np.ndarray, sigma: float=1.0, half_pixel_size: int = 1):
    """Create image from set of points, given an image shape"""
    img = np.zeros(shape, dtype=bool)
    for i in range(points.shape[0]):
        xr = int(points[i,0])
        if xr<0:
            xr=0
        if (xr)>img.shape[0]:
            xr=img.shape[0]-1
        yr = int(points[i,1])
        if yr<0:
            yr=0
        if (yr)>=img.shape[1]:
            yr=img.shape[1]-1
        img[xr,yr] = True
    img = cv2.morphologyEx(src=img.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.disk(half_pixel_size)).astype(bool)
    img = cv2.GaussianBlur(img.astype(np.uint8)*255,ksize=[0,0],sigmaX=sigma)
    return img/np.max(img)*255


# from: https://stackoverflow.com/a/26392655
def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def angle_code_from_point_sequence(points: np.ndarray):
    """For a set of ordered points assign a code based on angle to following point"""
    angles = np.array([get_angle(points[j,:]-points[j+1,:],[0,0],[1,0]) for j in range(len(points)-1)])
    angdiffs = np.array([np.abs(angles-180), np.abs(angles-90), np.abs(angles), np.abs(angles+90), np.abs(angles+180)]).T
    angdiffsmin = np.min(angdiffs,axis=1)
    angle_codes = np.array([np.where(angdiffs[j,:]==angdiffsmin[j])[0][0] for j in range(len(angdiffsmin))])
    return "".join(list(np.array(["L","B","R","T","L"])[angle_codes]))

def indices_sequence_from_ordered_points(ind: int, nn1: int, nn2: int, max_len: int):
    """Create indices list for ordered sets"""
    if ind-nn1 <0:
        tmpinds = np.concatenate([np.arange((ind-nn1)%max_len,max_len),np.arange(0,ind+nn2+1)])
    elif ind+nn2 >= max_len:
        tmpinds = np.concatenate([np.arange(ind-nn1,max_len),np.arange(0,(ind+nn2)%max_len+1)])
    else:
        tmpinds = np.arange(ind-nn1,ind+nn2+1)
    return tmpinds


def concave_boundary_from_grid(points: np.ndarray, max_dist: float=1.1, max_angle_diff: float=9, direction=1, max_allowed_counter_steps: int=5, centroid = None):
    if points.shape[0]<5:
        return shapely.LineString(points)
    border_points = np.unique(np.array(shapely.geometry.Polygon(points).convex_hull.exterior.coords.xy).T, axis=0)
    init_point = border_points[border_points[:,0] == np.max(border_points[:,0]),:]
    init_point = init_point.flatten()
    init_point2 = init_point+np.array([1,0])
    kdt = KDTree(points, leaf_size=30, metric='euclidean')
    if centroid is None:
        centroid = np.array(shapely.geometry.Polygon(points).convex_hull.centroid.coords.xy).T[0]
    angle_to_init = [get_angle(init_point, centroid, init_point)]
    init_point_ind = kdt.query(init_point.reshape(1,-1), k=1, return_distance=False)[0][0]
    boundary_points=[init_point_ind]
    possible_angles = np.array([-180,-90,0,90,180])
    while True:
        distances, indices = kdt.query(init_point.reshape(1,-1), k=5, return_distance=True)
        nis = indices[0,1:][distances[0,1:] < max_dist]
        if len(nis)==0:
            break
        angles = np.array([get_angle(init_point2,init_point,points[j,:]) for j in nis])
        absangles = np.abs(np.array(angles))
        to_keep_angle = np.logical_or(
                np.logical_or(absangles < max_angle_diff, absangles > 180-max_angle_diff),
                np.logical_and(absangles > 90-max_angle_diff, absangles < 90+max_angle_diff),
        )
        angles = angles[to_keep_angle]
        if len(angles)==0:
            break
        angdiffs = np.array([np.abs(possible_angles-a) for a in angles])
        angdiffsmin = np.min(angdiffs,axis=1)
        angle_codes = np.array([np.where(angdiffs[j,:]==angdiffsmin[j])[0][0] for j in range(len(angdiffsmin))])
        angles_corrected = possible_angles[angle_codes]
        angles_corrected[angles_corrected==-180]=180
        angles_corrected[angles_corrected==-90]=270
        if direction==1:
            angles_corrected[angles_corrected==0]=360
            ind1 = angles_corrected == np.min(angles_corrected)
        else:
            ind1 = angles_corrected == np.max(angles_corrected)

        next_ind = nis[to_keep_angle][ind1][0]
        angle_to_init_tmp = get_angle(init_point, centroid, points[next_ind,:])
        # full rotation
        if next_ind == boundary_points[0]:
            boundary_points.append(next_ind)
            # print("b1")
            break
        # other
        if next_ind in np.array(boundary_points):

            # check if not palindrome, i.e. linear out and back
            tls = boundary_points + [next_ind]
            if not np.any(np.array([tls[-i:]==tls[-i:][::-1] for i in range(3,len(boundary_points)+2,2)])):
                # print("b2")
                break 

        if np.sum(np.array(boundary_points) == next_ind)>1:
            # print("b3")
            break

        if len(angle_to_init)>=max_allowed_counter_steps:
            if direction==1:
                nums = np.array(angle_to_init)<0
            else:
                nums = np.array(angle_to_init)>0

            import itertools
            max_cons_wrong_angles = [len(list(y)) for (c,y) in itertools.groupby(nums) if c]
            if len(max_cons_wrong_angles)>0:
                max_cons_wrong_angles=max(max_cons_wrong_angles)
            else:
                max_cons_wrong_angles=0

            if max_cons_wrong_angles >= max_allowed_counter_steps:
                # print("b4")
                break
        init_point2=init_point
        init_point=points[next_ind,:]
        angle_to_init.append(angle_to_init_tmp)
        boundary_points.append(next_ind)

   
    # if boundary_points[0]!=boundary_points[-1] or len(boundary_points)==1:
    #     boundary_points.append(centroid_ind)
    #     boundary_points.append(boundary_points[0])

    pts = points[np.array(boundary_points),:]
    if len(boundary_points)==1:
        return shapely.LineString()
    if len(boundary_points)<4:
        return shapely.LineString(pts)

    tpts = points[np.array(boundary_points[-2:]),:]
    tpts[1]+=(tpts[0]-tpts[1])*1e-3
    t1=shapely.geometry.LineString(tpts)
    t2=shapely.geometry.LineString(points[np.array(boundary_points[1:-2]),:])
    inters1 = shapely.intersects(t1,t2)
    tpts = points[np.array(boundary_points[-3:-1]),:]
    tpts[0]+=(tpts[1]-tpts[0])*1e-3
    t1=shapely.geometry.LineString(tpts)
    t2=shapely.geometry.LineString(points[np.array(boundary_points[:-3]),:])
    inters2 = shapely.intersects(t1,t2)
    does_intersect = np.any(np.array([inters1,inters2]))
    i=1
    while does_intersect:
        tmpb = boundary_points[:len(boundary_points)-2-i] + boundary_points[-2:]
        pts = points[np.array(tmpb),:]
        if len(tmpb)==4:
            # print("b5")
            break
        i+=1
        tpts = points[np.array(tmpb[-2:]),:]
        tpts[1]+=(tpts[0]-tpts[1])*1e-3
        t1=shapely.geometry.LineString(tpts)
        t2=shapely.geometry.LineString(points[np.array(tmpb[1:-2]),:])
        inters1 = shapely.intersects(t1,t2)
        tpts = points[np.array(tmpb[-3:-1]),:]
        tpts[0]+=(tpts[1]-tpts[0])*1e-3
        t1=shapely.geometry.LineString(tpts)
        t2=shapely.geometry.LineString(points[np.array(tmpb[:-3]),:])
        inters2 = shapely.intersects(t1,t2)
        does_intersect = np.any(np.array([inters1,inters2]))

    if np.all(pts[0]==pts[-1]):
        po = shapely.Polygon(pts)
    else:
        po = shapely.LineString(pts)
    return po

def concave_boundary_from_grid_holes(points: np.ndarray, max_dist: float=1.4, max_angle_diff: float=25, max_allowed_counter_steps: int=5, centroid = np.array([0,0])):
    polyout = shapely.LineString()
    centroid = np.array(shapely.geometry.Polygon(points).convex_hull.centroid.coords.xy).T[0]
    pinit = shapely.Polygon(points).convex_hull
    pinit1 = pinit.buffer(1)
    pinit2 = pinit.buffer(-5)
    tpls_all = [shapely.geometry.Point(points[i,:]) for i in range(points.shape[0])]
    pconts1 = np.array([pinit1.contains_properly(tpls_all[i]) for i in range(len(tpls_all))])
    pconts2 = np.array([pinit2.contains_properly(tpls_all[i]) for i in range(len(tpls_all))])
    points = points[np.logical_and(pconts1,~pconts2)]

    while points.shape[0]>0:
        p1 = concave_boundary_from_grid(points,max_dist=max_dist, max_angle_diff=max_angle_diff, max_allowed_counter_steps=max_allowed_counter_steps, centroid=centroid)
        p2 = concave_boundary_from_grid(points,max_dist=max_dist, max_angle_diff=max_angle_diff,direction=2,max_allowed_counter_steps=max_allowed_counter_steps, centroid=centroid)
        if p1.is_empty and p2.is_empty:
            border_points = np.unique(np.array(shapely.geometry.Polygon(points).convex_hull.exterior.coords.xy).T, axis=0)
            init_point = border_points[border_points[:,0] == np.max(border_points[:,0]),:]
            init_point = init_point.flatten()
            kdt = KDTree(points, leaf_size=30, metric='euclidean')
            init_point_ind = kdt.query(init_point.reshape(1,-1), k=1, return_distance=False)[0][0]
            points = np.delete(points, init_point_ind, axis=0)
        else: 
            if p1.geom_type == "Polygon":
                c1 = p1.exterior.coords.xy
            else: 
                c1 = p1.xy
            if p2.geom_type == "Polygon":
                c2 = p2.exterior.coords.xy
            else: 
                c2 = p2.xy

            if len(c1[0]) > len(c2[0]):
                lcoords=np.array(c1).T
            else:
                lcoords=np.array(c2).T

            if polyout.is_empty:
                polyout = shapely.LineString(lcoords)
                newcoords=lcoords
            else:
                refcoords = np.array(polyout.xy).T
                kdt = KDTree(refcoords, leaf_size=30, metric='euclidean')
                distances, indices = kdt.query(lcoords[[0,-1],:], k=1, return_distance=True)
                nearind = np.array([0,-1])[(distances==np.min(distances)).reshape(1,-1)[0]][0]
                refind = indices[(distances==np.min(distances)).reshape(1,-1)[0]][0][0]
                # print(f"nearind: {nearind}, refind: {refind}")
                if nearind == -1:
                    if refind < refcoords.shape[0]/2:
                        newcoords = np.concatenate([lcoords,refcoords])
                    else: 
                        newcoords = np.concatenate([refcoords,lcoords[::-1]])
                else:
                    if refind < refcoords.shape[0]/2:
                        newcoords = np.concatenate([lcoords[::-1],refcoords])
                    else:
                        newcoords = np.concatenate([refcoords,lcoords])
                polyout = shapely.LineString(newcoords)
                # shapely.plotting.plot_line(polyout)
                # plt.show()

            tr = 20 if len(newcoords)>=40 else len(newcoords)
            angles = np.array([get_angle(newcoords[i,:], centroid, newcoords[-j,:]) for i in range(tr) for j in range(tr)])
            angles[angles<=0]=-angles[angles<=0]+180
            indm = angles.argmax()
            j=int(indm%tr)
            i=int(indm//tr)
            get_angle(newcoords[i,:], centroid, newcoords[-j,:])
            np.max(angles)
            if j == 0:
                newcoords_filt = newcoords[i:,:]
            else:
                newcoords_filt = newcoords[i:-j,:]

            po = shapely.Polygon(np.concatenate([newcoords_filt,centroid.reshape(1,-1)]))
            # shapely.plotting.plot_polygon(po)
            # plt.show()
            pot = po.buffer(0.1, cap_style='square', join_style='bevel')
            tpls_all = [shapely.geometry.Point(points[i,:]) for i in range(points.shape[0])]
            pconts1 = np.array([pot.contains_properly(tpls_all[i]) for i in range(len(tpls_all))])
            points = points[~pconts1]
    return polyout

