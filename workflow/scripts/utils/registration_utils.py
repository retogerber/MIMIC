import numpy as np
from typing import Union
import cv2
import skimage
import SimpleITK as sitk
import shapely
import sklearn

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

def dist_centroids(cent1, cent2, rescale):
    '''Calculate euclidean distance between centroids'''
    euclid_dist_pixel = ((cent1[0]-cent2[0])**2 + (cent1[1]-cent2[1])**2)**0.5
    euclid_dist = euclid_dist_pixel*rescale
    return euclid_dist



def create_ring_mask(img: np.ndarray, outscale: int, inscale: int) -> np.ndarray:
    '''
    Create a ring-shaped mask from an binary input image.

    Parameters:
    img (np.ndarray): The input image. It should be a 2D boolean array.
    outscale (int): The number of pixels to scale outwards from the current border when creating the outer mask.
    inscale (int): The number of pixels to scale inwards from the current border when creating the inner mask.

    Returns:
    ringmask (np.ndarray): The ring-shaped mask.
    '''
    outermask = cv2.morphologyEx(src=img.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int(outscale))).astype(bool)
    tmmask = np.zeros((img.shape[0]+2,img.shape[1]+2))
    tmmask[1:(img.shape[0]+1),1:(img.shape[1]+1)] = img 

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


def create_imz_coords(imzimg: np.ndarray, mask: Union[None, np.ndarray], imzrefcoords: np.ndarray, bbox: list, rotmat: np.ndarray) -> np.ndarray:
    '''
    Create coordinate matrices for image registration.

    This function takes an image, an optional mask, reference coordinates, a bounding box, and a rotation matrix as input.
    It creates coordinate matrices for the image, applies the mask if provided, and adjusts the coordinates based on the bounding box.
    The function then rotates the reference coordinates using the rotation matrix.
    Finally, it filters the image coordinates to only include those that are also in the reference coordinates.

    Parameters:
    imzimg (np.ndarray): The input image.
    mask (Union[None, np.ndarray]): An optional mask to apply to the image.
    imzrefcoords (np.ndarray): The reference coordinates for image registration.
    bbox (list): The bounding box to apply to the image coordinates.
    rotmat (np.ndarray): The rotation matrix to apply to the reference coordinates.

    Returns:
    imzcoords (np.ndarray): The filtered image coordinates.
    '''
    # create coordsmatrices for IMS
    indmaty, indmatx = np.indices(imzimg.shape)
    indmatx = indmatx.astype(np.uint32)
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


def image_from_points(shape, points: np.ndarray, sigma: float=1.0, half_pixel_size: int = 1)->np.ndarray:
    """Create image from set of points, given an image shape"""
    img = np.zeros(shape, dtype=np.uint8)
    for i in range(points.shape[0]):
        xr, yr = np.clip(points[i].astype(int), 0, np.array(shape)-1)
        img[xr,yr] = 255
    cv2.morphologyEx(src=img, dst=img, op = cv2.MORPH_DILATE, kernel = skimage.morphology.disk(half_pixel_size))
    cv2.GaussianBlur(src=img, dst=img,ksize=[0,0],sigmaX=sigma)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img


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

# adapted from: https://stackoverflow.com/a/26392655
def get_angle_vec(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    return np.degrees(np.arctan2(np.linalg.det(np.stack((v0,np.tile(v1, (v0.shape[0], v0.shape[1], 1))),axis=-1)),np.dot(v0,v1)))


def angle_code_from_point_sequence(points: np.ndarray)->str:
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


def concave_boundary_from_grid(points: np.ndarray, max_dist: float=1.1, max_angle_diff: float=9, direction=1, max_allowed_counter_steps: int=5, centroid = None, init_point = None, which_border="right"):
    """
    boundary from regular grid points
    
    points: 2D numpy array of points
    max_dist: maximum distance to admit as neighbors
    max_angle_diff: maximum deviation (in either direction) of angle from theoretical values (0,90,180,270)
    direction: 1 (anti-clockwise ) or 2 (clockwise)
    max_allowed_counter_steps: maximum number of steps going in opposite direction before stopping
    centroid: centroid position of points
    maxit: maximum number of iterations before stopping
    init_point: initial point on boundary
    which_border: location of initial_point
    """
    if points.shape[0]<5:
        return shapely.LineString(points)
    
    # setup initial points
    tmpch = shapely.geometry.Polygon(points).convex_hull
    if tmpch.geom_type == "Polygon":
        border_points = np.unique(np.array(tmpch.exterior.coords.xy).T, axis=0)
    else:
        return shapely.LineString(points)

    if init_point is None:
        init_point = border_points[border_points[:,0] == np.max(border_points[:,0]),:]
        init_point = init_point.flatten()
    if which_border=="right":
        init_point2 = init_point+np.array([1,0])
    elif which_border=="top":
        init_point2 = init_point+np.array([0,1])
    elif which_border=="left":
        init_point2 = init_point+np.array([-1,0])
    else:
        init_point2 = init_point+np.array([0,-1])
    kdt = sklearn.neighbors.KDTree(points, leaf_size=30, metric='euclidean')
    if centroid is None:
        centroid = np.array(shapely.geometry.Polygon(points).convex_hull.centroid.coords.xy).T[0]
    angle_to_init = [get_angle(init_point, centroid, init_point)]
    init_point_ind = kdt.query(init_point.reshape(1,-1), k=1, return_distance=False)[0][0]
    boundary_points=[init_point_ind]
    possible_angles = np.array([-180,-90,0,90,180])
    while True:
        # find neighbors
        distances, indices = kdt.query(init_point.reshape(1,-1), k=5, return_distance=True)
        nis = indices[0,1:][distances[0,1:] < max_dist]

        # no neighbors
        if len(nis)==0:
            break

        # filter neighbors by angle
        angles = np.array([get_angle(init_point2,init_point,points[j,:]) for j in nis])
        absangles = np.abs(np.array(angles))
        to_keep_angle = np.logical_or(
                np.logical_or(absangles < max_angle_diff, absangles > 180-max_angle_diff),
                np.logical_and(absangles > 90-max_angle_diff, absangles < 90+max_angle_diff),
        )
        angles = angles[to_keep_angle]

        # no neighbors with matching angle
        if len(angles)==0:
            break

        # correct angle to theoretical value
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

        # if chosen point is alread in list 
        if next_ind in np.array(boundary_points):

            # if total sum is close to 360, means full circle
            angle_total_sum = np.abs(np.sum(angle_to_init))
            # check if not palindrome, i.e. linear out and back
            tls = boundary_points + [next_ind]

            # not full circle (around centroid), but small loop
            if not np.any(np.array([tls[-i:]==tls[-i:][::-1] for i in range(3,len(boundary_points)+2,2)])) and angle_total_sum < 350:
                break 

        # chosen point already in set more than once (e.g. if already once there allowed to add another time)
        if np.sum(np.array(boundary_points) == next_ind)>1:
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

            # if too many steps in wrong direction
            if max_cons_wrong_angles >= max_allowed_counter_steps:
                break

        # add points
        angle_to_init.append(angle_to_init_tmp)
        boundary_points.append(next_ind)
        # prepare for next iteration
        init_point2=init_point
        init_point=points[next_ind,:]

    pts = points[np.array(boundary_points),:]
    # if full circle
    if len(boundary_points)==1:
        return shapely.LineString()
    elif len(boundary_points)<5:
        return shapely.LineString(pts)
    elif boundary_points[0]==boundary_points[-1]:
        return shapely.Polygon(pts)

    # try to remove intersections
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

def concave_boundary_from_grid_holes(points: np.ndarray, max_dist: float=1.4, max_angle_diff: float=25, max_allowed_counter_steps: int=5, centroid = None, maxit: int=1000, direction=1):
    """
    boundary from regular grid points with missing points

    points: 2D numpy array of points
    max_dist: maximum distance to admit as neighbors
    max_angle_diff: maximum deviation (in either direction) of angle from theoretical values (0,90,180,270)
    max_allowed_counter_steps: maximum number of steps going in opposite direction before stopping
    centroid: centroid position of points
    maxit: maximum number of iterations before stopping
    direction: 1 (anti-clockwise ) or 2 (clockwise)
    """

    polyout = shapely.LineString()
    all_points = points.copy()
    if centroid is None:
        centroid = np.array(shapely.geometry.Polygon(points).convex_hull.centroid.coords.xy).T[0]
    # filter points to contain only points at boundary (for speedup)
    pinit = shapely.Polygon(points).convex_hull
    pinit1 = pinit.buffer(max_dist)
    pinit2 = pinit.buffer(-10*max_dist)
    tpls_all = [shapely.geometry.Point(points[i,:]) for i in range(points.shape[0])]
    pconts1 = np.array([pinit1.contains_properly(tpls_all[i]) for i in range(len(tpls_all))])
    pconts2 = np.array([pinit2.contains_properly(tpls_all[i]) for i in range(len(tpls_all))])
    points = points[np.logical_and(pconts1,~pconts2)]

    # set initial starting point (right border)
    border_points = np.unique(np.array(shapely.geometry.Polygon(points).convex_hull.exterior.coords.xy).T, axis=0)
    tmpb = border_points[border_points[:,0] > (np.max(border_points[:,0])-1),:]
    global_init_point = tmpb[tmpb[:,1]==np.min(tmpb[:,1])][0]

    itersnpoints = []
    # iterate
    iter = 0
    while points.shape[0]>1 and iter<maxit:
        iter+=1
        # get new initial point, closest (angle wise) to global initial point
        tmppoints = np.concatenate([points,centroid.reshape(1,-1)])
        border_points = np.unique(np.array(shapely.geometry.Polygon(tmppoints).convex_hull.exterior.coords.xy).T, axis=0)
        wcent = np.sum(border_points == centroid,axis=1).argmax()
        border_points = np.delete(border_points,wcent,axis=0)
        angles_points = np.array([get_angle(global_init_point, centroid, border_points[i,:]) for i in range(border_points.shape[0])])
        if direction==1:
            angles_points=angles_points%360
        else:
            angles_points=-angles_points%360
        angles_points[angles_points==360]=0
        init_point = border_points[angles_points == np.min(angles_points),:][0]
        init_point_angle = angles_points[angles_points == np.min(angles_points)][0]
        # get direction of second initial point
        if init_point_angle<45:
            which_border="right"
        elif init_point_angle<135:
            which_border="top" if direction==1 else "bottom"
        elif init_point_angle<225:
            which_border="left"
        elif init_point_angle<315:
            which_border="bottom" if direction==1 else "top"
        else:
            which_border="right"
        
        # find points 
        p1 = concave_boundary_from_grid(
            points,
            max_dist=max_dist, 
            max_angle_diff=max_angle_diff,
            max_allowed_counter_steps=max_allowed_counter_steps,
            centroid=centroid,
            init_point=init_point,
            which_border=which_border, 
            direction=direction)

        # shapely.plotting.plot_line(p1)
        # plt.scatter(points[:,0],points[:,1],alpha=0.1)
        # plt.scatter(border_points[:,0],border_points[:,1])
        # plt.scatter(init_point[0],init_point[1], c="red")
        # plt.scatter(global_init_point[0],global_init_point[1], c="blue")
        # plt.show()

        # if no neighboring points to initial point are found, remove inital point
        if p1.is_empty:
            kdt = sklearn.neighbors.KDTree(points, leaf_size=30, metric='euclidean')
            init_point_ind = kdt.query(init_point.reshape(1,-1), k=1, return_distance=False)[0][0]
            points = np.delete(points, init_point_ind, axis=0)
        else: 
            if p1.geom_type == "Polygon":
                lcoords = np.array(p1.exterior.coords.xy).T
            else: 
                lcoords = np.array(p1.xy).T

            # first iteration create object
            if polyout.is_empty:
                polyout = shapely.LineString(lcoords)
                newcoords=lcoords
            else:
                # add new points to existing points
                refcoords = np.array(polyout.xy).T
                angles_points = np.array([get_angle(global_init_point, centroid, refcoords[i,:]) for i in range(refcoords.shape[0])])
                if direction==1:
                    angles_points=angles_points%360
                else:
                    angles_points=-angles_points%360
                if direction==2 and len(refcoords)<40:
                    angles_points[angles_points>270]=-1
                angles_ind = angles_points > np.quantile(angles_points,0.55)
                tmprefcoords = refcoords[angles_ind,]
                angles_ls = []
                for i in range(tmprefcoords.shape[0]): 
                    for j in range(lcoords.shape[0]):
                        tmpang = get_angle(tmprefcoords[i,:], centroid, lcoords[j,:])
                        tmpang = tmpang%360 if direction==1 else -tmpang%360
                        angles_ls.append([tmpang,i,j])
                angles_mat=np.array(angles_ls)
                indm = angles_mat[:,0].argmin()
                i = int(angles_mat[indm,1])
                j = int(angles_mat[indm,2])
                i = np.arange(refcoords.shape[0])[angles_ind][i]
                # refcoords[i,:]
                # lcoords[j,:]
                # get_angle(refcoords[i,:], centroid, lcoords[j,:])
                
                newcoords = np.concatenate([refcoords,refcoords[i:,:][::-1],lcoords[:(j+1),:][::-1],lcoords])
                # tmppolyout = shapely.LineString(newcoords)
                # shapely.plotting.plot_line(tmppolyout)
                # plt.show()
                polyout = shapely.LineString(newcoords)


            # filter points
            # find points spanning maximum angle
            tr = int(newcoords.shape[0]/4-5) if len(newcoords)>=100 else int(len(newcoords))

            # small number of points
            if tr==newcoords.shape[0]:
                angles_ls = []
                for i in range(newcoords.shape[0]): 
                    for j in range(newcoords.shape[0]):
                        tmpang = get_angle(newcoords[i,:], centroid, newcoords[j,:])
                        tmpang = tmpang%360 if direction==1 else -tmpang%360
                        angles_ls.append([tmpang,i,j])
                angles_mat=np.array(angles_ls)
                angles_mat[angles_mat[:,0]>180,0]=-angles_mat[angles_mat[:,0]>180,0]%360
                indm = angles_mat[:,0].argmax()
                i = int(angles_mat[indm,1])
                j = int(angles_mat[indm,2])
                if i>j:
                    tt=j
                    j=i
                    i=tt
                newcoords_filt = newcoords[i:(j+1),:]
            else:
                tmpc1 = newcoords[:tr,:]
                tmpc2 = newcoords[-tr:,:][::-1]

                angles_ls = []
                for i in range(tmpc1.shape[0]): 
                    for j in range(tmpc2.shape[0]):
                        tmpang = get_angle(tmpc1[i,:], centroid, tmpc2[j,:])
                        tmpang = tmpang%360 if direction==1 else -tmpang%360
                        angles_ls.append([tmpang,i,j])
                angles_mat=np.array(angles_ls)
                indm = angles_mat[:,0].argmax()
                i = int(angles_mat[indm,1])
                j = int(angles_mat[indm,2])
                if j == 0:
                    newcoords_filt = newcoords[i:,:]
                else:
                    newcoords_filt = newcoords[i:(-j),:]

            # create new polygon
            tmp = np.concatenate([newcoords_filt,centroid.reshape(1,-1),newcoords_filt[0,:].reshape(1,-1)])
            # scale out
            tmp2 = tmp+0.5*(tmp-centroid)
            po = shapely.Polygon(tmp2).buffer(0.2)

            # import shapely.plotting
            # shapely.plotting.plot_line(polyout)
            # shapely.plotting.plot_polygon(po)
            # plt.scatter(points[:,0],points[:,1],alpha=0.1)
            # plt.scatter(newcoords_filt[0,0],newcoords_filt[0,1],c="red",alpha=0.5)
            # plt.scatter(newcoords_filt[-1,0],newcoords_filt[-1,1],c="blue",alpha=0.5)
            # plt.show()

            # filter points
            tpls_all = [shapely.geometry.Point(points[i,:]) for i in range(points.shape[0])]
            pconts1 = np.array([po.contains(tpls_all[i]) for i in range(len(tpls_all))])
            points = points[~pconts1]

        itersnpoints.append(points.shape[0])
        print(f"{iter}: n_points: {points.shape[0]:5}")

        if len(itersnpoints)>2:
            if np.all(np.diff(itersnpoints[-2:])==0):
                print("no change in number of points")
                break
    
    if iter >= maxit:
        # return alpha hull
        polyout = shapely.concave_hull(shapely.geometry.MultiPoint(all_points), ratio=0.01)
    
    refcoords = np.array(polyout.xy).T
    # if last point in list is not first point
    if (refcoords[0,0]!=refcoords[-1,0]) and (refcoords[0,1]!=refcoords[-1,1]):
        # combine points
        ref_ind1 = np.arange(refcoords.shape[0]) > refcoords.shape[0]*0.9
        tmprefcoords1 = refcoords[ref_ind1,:]
        ref_ind2 = np.arange(refcoords.shape[0]) < refcoords.shape[0]*0.1
        tmprefcoords2 = refcoords[ref_ind2,:]

        # filter by distance 
        kdt = sklearn.neighbors.KDTree(tmprefcoords1, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(tmprefcoords2, k=1)
        indices_filt = indices[distances==np.min(distances)]

        ref_ind11 = np.array([k in indices_filt for k in np.arange(len(tmprefcoords1))])
        ref_ind22 = np.array([k in indices_filt for k in indices])

        tmprefcoords1=tmprefcoords1[ref_ind11]
        tmprefcoords2=tmprefcoords2[ref_ind22]

        # find points from start and end with minimal angle
        angles_ls = []
        for i in range(tmprefcoords1.shape[0]): 
            for j in range(tmprefcoords2.shape[0]):
                tmpang = get_angle(tmprefcoords1[i,:], centroid, tmprefcoords2[j,:])
                tmpang = tmpang%360 if direction==1 else -tmpang%360
                angles_ls.append([tmpang,i,j])
        angles_mat=np.array(angles_ls)
        indm = angles_mat[:,0].argmin()
        i = int(angles_mat[indm,1])
        i = np.arange(refcoords.shape[0])[ref_ind1][ref_ind11][i]
        j = int(angles_mat[indm,2])
        j = np.arange(refcoords.shape[0])[ref_ind2][ref_ind22][j]

        # reshuffle points
        newcoords = np.concatenate([refcoords[:(j+1),:][::-1],refcoords,refcoords[i:,::][::-1],])
        polyout = shapely.Polygon(newcoords)
    else:
        polyout = shapely.Polygon(refcoords)


    return polyout

