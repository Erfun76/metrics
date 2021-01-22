from skimage.measure import compare_ssim

def calculate_ssim(p0, p1, range=255.):
    return compare_ssim(p0, p1, data_range=range, multichannel=True)