"""Exact discrete 90-degree rotation augmentation (no interpolation).

Mask2Former's RGB-D mapper needs a rotation that is exactly correct for a 6-channel
float32 depth-geometry map (height/normals/discontinuity/valid), not just RGB uint8. cv2's
warpAffine (used by detectron2's general-angle RotationTransform) is channel-limited and,
for an arbitrary angle, would resample/interpolate depth across occlusion boundaries — exactly
what rgbd_seg/data.py's RGBDMapper already avoids by using nearest resampling. Restricting to
multiples of 90 degrees lets us use an exact `np.rot90` permutation instead: every output pixel
is some input pixel, unchanged, for any number of channels and any dtype.
"""
import numpy as np
from detectron2.data.transforms import Augmentation, Transform


class Rot90Transform(Transform):
    """Rotate HxW or HxWxC arrays by k*90 degrees counter-clockwise (np.rot90 convention)."""

    def __init__(self, k, h, w):
        super().__init__()
        self.k = k % 4
        self.h, self.w = h, w

    def apply_image(self, img):
        return np.ascontiguousarray(np.rot90(img, self.k, axes=(0, 1)))

    def apply_segmentation(self, segmentation):
        return self.apply_image(segmentation)

    def apply_coords(self, coords):
        # coords: Nx2 continuous (x, y); detectron2 convention uses the un-offset image extent
        # (W, H), not (W-1, H-1) -- see HFlipTransform's docstring for the same convention.
        coords = np.asarray(coords, dtype=np.float32).copy()
        x, y = coords[:, 0].copy(), coords[:, 1].copy()
        h, w = self.h, self.w
        if self.k == 0:
            nx, ny = x, y
        elif self.k == 1:
            nx, ny = y, w - x
        elif self.k == 2:
            nx, ny = w - x, h - y
        else:  # k == 3
            nx, ny = h - y, x
        coords[:, 0], coords[:, 1] = nx, ny
        return coords

    def inverse(self):
        new_h, new_w = (self.w, self.h) if self.k % 2 else (self.h, self.w)
        return Rot90Transform((4 - self.k) % 4, new_h, new_w)


class RandomRot90(Augmentation):
    """Uniformly pick a rotation from `choices` (multiples of 90)."""

    def __init__(self, choices=(0, 90, 180, 270)):
        super().__init__()
        assert all(c % 90 == 0 for c in choices)
        self.choices = choices

    def get_transform(self, image):
        k = int(self.choices[np.random.randint(len(self.choices))]) // 90
        h, w = image.shape[:2]
        return Rot90Transform(k, h, w)
