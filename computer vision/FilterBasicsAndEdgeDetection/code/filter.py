from PIL import Image # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr, rescale='minmax'):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel 
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################
def sharpen(img, sigma, alpha):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    # Perform Gaussian smoothing
    blurred = ndimage.gaussian_filter(img, sigma=sigma)
    
    # Compute detailed image 
    detailed = img - blurred
    
    # Compute sharpened image
    sharpened = img + alpha * detailed
    
    # Truncate the values to [0, 255]
    sharpened = np.clip(sharpened, 0, 255)

    return sharpened

def median_filter(img, s):
    '''Perform median filter of size s x s to image 'arr', and return the filtered image.'''
    
    H, W, C = img.shape
    out = np.zeros_like(img)
    
    # Apply median filter to each channel separately 
    for c in range(C):
        for i in range(s//2, H-s//2):
            for j in range(s//2, W-s//2):
                neighbors = []
                for m in range(i-s//2, i+s//2+1):
                    for n in range(j-s//2, j+s//2+1):
                        neighbors.append(img[m,n,c])
                out[i,j,c] = np.median(neighbors)

    return out

if __name__ == '__main__':
    input_path = '../data/rain.jpeg'
    img = read_img_as_array(input_path)
    show_array_as_img(img)
    
    # 1.1 Gaussian blur
    blurred = ndimage.gaussian_filter(img, sigma=[3,3,0]) 
    save_array_as_img(blurred, '../data/1.1_blur.jpg')
    
    # 1.2 Sharpening
    sharpened = sharpen(img, sigma=3, alpha=3)  
    save_array_as_img(sharpened, '../data/1.2_sharpened.jpg')
    
    # 1.3 Median filter for derain
    derained = median_filter(img, s=5)
    save_array_as_img(derained, '../data/1.3_derained.jpg')
