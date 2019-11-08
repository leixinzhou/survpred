import numpy as np
from skimage.measure import find_contours
from matplotlib import patches, lines
from matplotlib.patches import Polygon

def show_img_contour(image, mask, ax, color):
    # masked_image = apply_mask(image, mask, color)
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5, 'high', 'high')
    ax.imshow(image, cmap='gray')
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax.add_patch(p)

