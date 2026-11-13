"""
Atlas region of source-space vertices on SUBJECT_TO.
Usage: python lookup_vertex_region.py --vertices 3387 2914 --max-dist 5
(the numbers shown as "Vertex" in the volume plots; without --vertices, VERTICES is used)

On a 10 mm grid a vertex often sits in a sulcus, white matter or at the brain edge, so the
nearest region within max-dist mm is reported instead of only the label at the vertex itself.
"""
import argparse
from pathlib import Path

import numpy as np
import mne
import nibabel as nib

from params import VOL_SPACING, SUBJECT_TO, fnames

VERTICES = [3387]   # default vertex numbers to look up
MAX_DIST_MM = 5     # default search radius; half the 10 mm grid spacing

# atlas labels that are not a brain region
NOT_A_REGION = ("Unknown", "White-Matter", "Ventricle", "CSF")


def vertex_region(vertex, src, atlas, id_to_name, max_dist=MAX_DIST_MM):
    """Nearest atlas region to one source-space vertex.
    Source positions are FreeSurfer surface RAS (m) -> mm -> atlas voxel via inverse vox2ras_tkr.
    Returns (region at the vertex, nearest region within max_dist mm or None, its distance in mm or None)."""
    pos_mm = src[0]["rr"][vertex] * 1000
    centre = np.round((np.linalg.inv(atlas.header.get_vox2ras_tkr()) @ np.r_[pos_mm, 1])[:3]).astype(int)
    zooms = np.array(atlas.header.get_zooms()[:3])

    # cube of atlas voxels around the vertex, then keep those within max_dist mm
    r = int(np.ceil(max_dist / zooms.min()))
    (i0, j0, k0), (i1, j1, k1) = centre - r, centre + r + 1
    ids = np.asarray(atlas.dataobj[i0:i1, j0:j1, k0:k1]).ravel()
    offsets = np.indices((2 * r + 1,) * 3).reshape(3, -1).T - r
    dist = np.linalg.norm(offsets * zooms, axis=1)

    at_vertex = id_to_name.get(int(ids[np.argmin(dist)]), "Unknown")

    is_region = np.array([not any(word in id_to_name.get(int(i), "Unknown") for word in NOT_A_REGION)
                          for i in ids])
    candidates = np.flatnonzero(is_region & (dist <= max_dist))
    if len(candidates) == 0:
        return at_vertex, None, None
    nearest = candidates[np.argmin(dist[candidates])]
    return at_vertex, id_to_name[int(ids[nearest])], float(dist[nearest])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas region of source-space vertices")
    parser.add_argument("--vertices", type=int, nargs="+", default=VERTICES,
                        help="vertex numbers shown as 'Vertex' in the volume plots")
    parser.add_argument("--max-dist", type=float, default=MAX_DIST_MM,
                        help="search radius in mm for the nearest region")
    args = parser.parse_args()

    src_to = mne.read_source_spaces(fnames.src_volume(subject=SUBJECT_TO, vol_spacing=VOL_SPACING))
    atlas = nib.load(Path(fnames.subjects_dir) / SUBJECT_TO / "mri" / "aparc+aseg.mgz")
    lut, _ = mne.read_freesurfer_lut()
    id_to_name = {v: k for k, v in lut.items()}

    for vertex in args.vertices:
        if not src_to[0]["inuse"][vertex]:
            print(f"{vertex}: not a vertex in use in the {SUBJECT_TO} source space")
            continue
        at_vertex, nearest, dist = vertex_region(vertex, src_to, atlas, id_to_name, args.max_dist)
        if nearest is None:
            print(f"{vertex}: no region within {args.max_dist:g} mm (at vertex: {at_vertex})")
        else:
            print(f"{vertex}: {nearest} ({dist:.1f} mm; at vertex: {at_vertex})")