"""502 label-free wafer features for a frozen supervised feature ablation.

The first 168 values exactly reproduce spatial_features. Added channels capture
radial/angular interactions, background-subtracted line projections, local
density contrast and positions/shapes of two largest components. None changes
the wafer or suppresses Random observations. An attractive long line can
coexist with a compact blob; the features do not force one dominant pattern.

Projection histograms sum failed dies along strips, inspired by the Radon
transform, but are not an interpolated/continuous Radon transform or a fitted
line detector. Official definition/reference:
https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html
No scikit-image dependency is used. Component elongation uses covariance
moments, not a claimed skeleton reconstruction.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

from .baseline import spatial_features

FEATURE_VERSION = "spatial_polar_line_density_components_v1"
FEATURE_GROUPS = {
    "spatial_v1": (0, 168),
    "polar_joint": (168, 392),
    "line_projection": (392, 446),
    "multiscale_density": (446, 470),
    "component_shape": (470, 502),
}
FEATURE_COUNT = 502
POLAR_EDGES = (
    (0., .25, .5, .7, .82, .92, 1.5),
    (0., .2, .4, .6, .75, .85, .93, 1.02, 1.5),
)
ANGLES_DEGREES = tuple(range(0, 180, 10))
_DENSITY_STATS = ("q25", "median", "q90", "std", "upper_decile_contrast",
                  "positive_residual_mean", "high_excess_fraction", "fail_local_mean")
_COMPONENT_STATS = ("area_fraction", "center_x", "center_y", "center_radius",
                    "eccentricity", "log_elongation", "bbox_fill", "perimeter_over_area")


def feature_names() -> list[str]:
    """Return one stable, unique name per output coordinate."""
    names = [f"spatial_v1_{i:03d}" for i in range(168)]
    for edges in POLAR_EDGES:
        rings = len(edges)-1
        names.extend(f"polar_{rings}x16_r{ring:02d}_a{angle:02d}" for ring in range(rings) for angle in range(16))
    names.extend(f"line_normal_{angle:03d}_{stat}" for angle in ANGLES_DEGREES
                 for stat in ("peak_z", "peak_excess_share", "excess_entropy"))
    names.extend(f"density_r{radius}_{stat}" for radius in (1,2,4) for stat in _DENSITY_STATS)
    names.extend(f"component_{kind}_rank{rank}_{stat}" for kind in ("raw_fail","dense_r2")
                 for rank in (1,2) for stat in _COMPONENT_STATS)
    return names


@lru_cache(maxsize=32)
def _geometry(height: int, width: int, y0: int, y1: int, x0: int, x1: int):
    yy,xx = np.indices((height,width),dtype=float)
    cy,cx = (y0+y1)/2,(x0+x1)/2
    ry,rx = max((y1-y0)/2,1),max((x1-x0)/2,1)
    y,x = (yy-cy)/ry,(xx-cx)/rx
    radial = np.hypot(x,y)
    angular = np.floor(((np.arctan2(y,x)+2*np.pi)%(2*np.pi))/(2*np.pi)*16).astype(np.int16)
    polar = []
    for edges in POLAR_EDGES:
        rbin = np.clip(np.searchsorted(edges,radial,side="right")-1,0,len(edges)-2)
        polar.append((rbin*16+angular).reshape(-1))
    projection = []
    for degrees in ANGLES_DEGREES:
        theta = np.deg2rad(degrees)
        coordinate = x*np.cos(theta)+y*np.sin(theta)
        # Normalized wafer coordinates fit [-sqrt(2),sqrt(2)]; 48 fixed bins
        # cover [-1.5,1.5], about 1.5 dies per strip for a 52x52 map.
        projection.append(np.clip(np.floor((coordinate+1.5)/3*48),0,47).astype(np.int16).reshape(-1))
    return x,y,polar,projection,1/min(rx,ry)


def _component_features(mask: np.ndarray, active_count: int, x: np.ndarray,
                        y: np.ndarray, coordinate_step: float) -> list[float]:
    labels,n = ndimage.label(mask,structure=np.ones((3,3),dtype=np.uint8))
    if not n:
        return [0.]*16
    sizes = np.bincount(labels.reshape(-1))[1:]
    order = np.argsort(-sizes,kind="stable")[:2]+1
    values = []
    for component in order:
        selected = labels==component
        rows,cols = np.nonzero(selected)
        count = len(rows)
        px,py = x[selected],y[selected]
        cx,cy = float(px.mean()),float(py.mean())
        if count>1:
            # Analytic 2x2 covariance avoids per-component np.cov allocations.
            dx,dy = px-cx,py-cy
            vx,vy,cov = np.mean(dx*dx),np.mean(dy*dy),np.mean(dx*dy)
            delta = float(np.hypot(vx-vy,2*cov))
            major,minor = max(float((vx+vy+delta)/2),0.),max(float((vx+vy-delta)/2),0.)
            eccentricity = np.sqrt(max(0.,1-minor/max(major,1e-12)))
            regularizer = coordinate_step**2/12
            log_elongation = np.log1p(np.sqrt((major+regularizer)/(minor+regularizer)))
        else:
            eccentricity,log_elongation = 0.,np.log(2.)
        # Four-neighbor exposed edges, including wafer boundaries. This is an
        # inexpensive discrete perimeter statistic, not continuous geometry.
        crop = np.pad(selected[rows.min():rows.max()+1,cols.min():cols.max()+1],1)
        center = crop[1:-1,1:-1]
        perimeter = int(np.sum(center&~crop[:-2,1:-1])+np.sum(center&~crop[2:,1:-1])+
                        np.sum(center&~crop[1:-1,:-2])+np.sum(center&~crop[1:-1,2:]))
        bbox = int((rows.max()-rows.min()+1)*(cols.max()-cols.min()+1))
        values.extend([count/active_count,cx,cy,np.hypot(cx,cy),eccentricity,
                       log_elongation,count/bbox,perimeter/count])
    values.extend([0.]*(16-len(values)))
    return values


def advanced_features(wafer) -> np.ndarray:
    """Extract 502 finite float32 features from a 0/1/2 two-dimensional map.

    Added feature densities/projections exclude outside dies from numerator
    and denominator. The inherited 168 include an explicit active-area/image-
    area geometry ratio; only that inherited convention sees exterior padding.
    The input array is never modified and no labels or split information enter.
    """
    wafer = np.asarray(wafer)
    if wafer.ndim!=2 or not np.isin(wafer,[0,1,2]).all():
        raise ValueError("Expected a 2D map with 0=outside, 1=pass, 2=fail")
    active,fail = wafer>0,wafer==2
    if not active.any():
        raise ValueError("A wafer map contains no active dies")
    row,col = np.nonzero(active)
    x,y,polar,projections,step = _geometry(*wafer.shape,int(row.min()),int(row.max()),int(col.min()),int(col.max()))
    aflat,fflat = active.reshape(-1),fail.reshape(-1)
    total = int(active.sum())
    global_density = float(fail.sum()/total)
    values = spatial_features(wafer).tolist()
    # Joint distributions retain the association of rim position with angle;
    # separate radial/angular marginals in the original baseline lose it.
    for ids,edges in zip(polar,POLAR_EDGES):
        count = (len(edges)-1)*16
        available = np.bincount(ids[aflat],minlength=count)
        failed = np.bincount(ids[fflat],minlength=count)
        density = np.divide(failed,available,out=np.zeros(count,dtype=float),where=available>0)
        values.extend(density)
    # Background-subtracted strip evidence: subtract global expected failures,
    # normalize by binomial variance, and describe concentration per direction.
    q = float(np.clip(global_density,.01,.99))
    for ids in projections:
        available = np.bincount(ids[aflat],minlength=48)
        failed = np.bincount(ids[fflat],minlength=48)
        eligible = available>=max(5.,.2*float(available.max()))
        excess = np.maximum(failed-q*available,0.)*eligible
        z = np.divide(excess,np.sqrt(available*q*(1-q)+1.),out=np.zeros(48),where=eligible)
        mass = float(excess.sum())
        if mass>0:
            share = excess/mass
            nonzero = share[share>0]
            entropy = float(-(nonzero*np.log(nonzero)).sum()/np.log(max(int(eligible.sum()),2)))
            values.extend([float(z.max()),float(share.max()),entropy])
        else:
            values.extend([0.,0.,0.])
    densities = {}
    for radius in (1,2,4):
        # uniform_filter is a separable box average; multiplying by kernel area
        # recovers counts, then dividing by valid counts handles mask boundaries.
        size = 2*radius+1
        failed_count = ndimage.uniform_filter(fail.astype(float),size=size,mode="constant",cval=0)*size**2
        available_count = ndimage.uniform_filter(active.astype(float),size=size,mode="constant",cval=0)*size**2
        local = np.divide(failed_count,available_count,out=np.zeros_like(failed_count),where=available_count>1e-9)
        local = np.clip(local,0.,1.)
        densities[radius] = local
        selected = local[active]
        q25,median,q90 = np.quantile(selected,[.25,.5,.9])
        background = float(np.median(selected[selected<=median]))
        residual = np.maximum(selected-background,0.)
        standard_error = np.sqrt(max(background*(1-background),.01)/np.maximum(available_count[active],1.))
        top = selected[selected>=q90]
        values.extend([q25,median,q90,float(selected.std()),float(top.mean()-median),
                       float(residual.mean()),float(np.mean(selected>background+2*standard_error)),
                       float(local[fail].mean()) if fail.any() else 0.])
    median = float(np.median(densities[2][active]))
    background = float(np.median(densities[2][active&(densities[2]<=median)]))
    dense_mask = active&(densities[2]>=max(.35,background+.15))
    for mask in (fail,dense_mask):
        values.extend(_component_features(mask,total,x,y,step))
    result = np.nan_to_num(np.asarray(values,dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
    if result.shape!=(FEATURE_COUNT,):
        raise RuntimeError(f"Feature schema mismatch: {result.shape}, expected ({FEATURE_COUNT},)")
    return result
