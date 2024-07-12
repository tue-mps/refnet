import numpy as np
import scipy as sp
import scipy.ndimage
# from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt


def main():
    im = Image.open('/home/kach271771/Documents/Matlab_BEV/matlab/dataset/RADIal_sample_Matlab/Matlab_output/image_000018.jpg')
    im = im.convert('RGB')
    data = np.array(im)

    plot_polar_image(data, origin=None)
    plt.show()


def plot_polar_image(data, origin=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    polar_grid, r, theta = reproject_image_into_polar(data, origin)
    # plt.figure()
    # plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
    # plt.axis('auto')

    fig = plt.figure(frameon=False)
    fig.set_size_inches(9.00, 5.40)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    # ax.set_ylim(ax.set_ylim()[::-1])
    fig.add_axes(ax)
    polar_grid = sp.ndimage.rotate(polar_grid, 360)

    ax.imshow(polar_grid, aspect='auto', extent=(theta.min(), theta.max(), r.max(), r.min()))
    fig.savefig("/home/kach271771/Documents/Matlab_BEV/matlab/dataset/RADIal_sample_Matlab/Python_output/image_000018.jpg")
    im = Image.open('/home/kach271771/Documents/Matlab_BEV/matlab/dataset/RADIal_sample_Matlab/Python_output/image_000018.jpg')
    im = im.convert('RGB')
    dataPolar = np.array(im)
    image_arr_topcrop = dataPolar[28:540, 0:900]
    image_arr_ccrop = image_arr_topcrop[0:540, 322:578]
    image_arr_ccrop = np.flip(image_arr_ccrop, axis=1)
    plt.imsave("/home/kach271771/Documents/Matlab_BEV/matlab/dataset/RADIal_sample_Matlab/Python_output/image_000018.jpg",image_arr_ccrop)


def cart2polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def polar2cart(r, theta):
    x = r * np.cos(theta) #change this sign to flip the image
    y = r * np.sin(theta)
    return x, y


def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        # origin = (nx//2, ny//2)
        origin = (nx // 2, 0)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin[0]
    y -= origin[1]

    # Determine that the min and max r and theta coords will be...
    # x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0]  # We need to shift the origin back to
    yi += origin[1]  # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi))  # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    bands = []
    for band in data.T:
        zi = sp.ndimage.map_coordinates(band, coords, order=1)
        bands.append(zi.reshape((nx, ny)))
    output = np.dstack(bands)
    return output, r_i, theta_i


if __name__ == '__main__':
    main()
