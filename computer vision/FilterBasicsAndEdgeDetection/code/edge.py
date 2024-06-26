from PIL import Image # pillow package 
import numpy as np
from scipy import ndimage
import math

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255
        
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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
    
def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''

    weights_x = [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]]
    weights_y = [[ 1, 2, 1],
                 [ 0, 0, 0],
                 [-1,-2,-1]]
    
    Gx = ndimage.convolve(arr, weights_x)
    Gy = ndimage.convolve(arr, weights_y)
    G = np.sqrt(np.square(Gx) + np.square(Gy)) 

    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    row_max, col_max= G.shape
    G_result = G.copy()

    for i in range(1, row_max-1):
        for j in range(1, col_max - 1):
            theta = np.arctan2(Gy[i,j], Gx[i,j]) * 180 / np.pi
            
            if (theta >= -22.5 and theta <= 22.5) or (theta >= 157.5 and theta <= 180) or (theta >= -180 and theta <= -157.5):
                N1 = G[i, j+1]  
                N2 = G[i, j-1]
            elif (theta >= 22.5 and theta <= 67.5) or (theta >= -157.5 and theta <= -112.5):
                N1 = G[i+1, j+1] 
                N2 = G[i-1, j-1]
            elif (theta >= 67.5 and theta <= 112.5) or (theta >= -112.5 and theta <= -67.5):
                N1 = G[i+1, j]
                N2 = G[i-1, j]
            elif (theta >= 112.5 and theta <= 157.5) or (theta >= -67.5 and theta <= -22.5):
                N1 = G[i-1, j+1] 
                N2 = G[i+1, j-1]
            
            if G[i,j] < N1 or G[i,j] < N2:
                G_result[i,j] = 0
            else:
                G_result[i,j] = G[i,j]

    return G_result

def thresholding(G, t):
    '''Binarize G according threshold t'''
    G_binary = G.copy()
    G_binary[G <= t] = 0
    G_binary[G > t] = 255
        
    return G_binary

def hysteresis_thresholding(G, low, high):
    G_low = thresholding(G, low)
    G_high = thresholding(G, high)
    
    G_hyst = G_high.copy()
    
    changed = True
    while changed:
        changed = False
        for i in range(1, G_hyst.shape[0]-1):
            for j in range(1, G_hyst.shape[1]-1):
                if G_low[i,j] == 255 and G_hyst[i,j] == 0:
                    if G_hyst[i-1,j-1]==255 or G_hyst[i-1,j]==255 or G_hyst[i-1,j+1]==255 or \
                       G_hyst[i,j-1]==255 or G_hyst[i,j+1]==255 or \
                       G_hyst[i+1,j-1]==255 or G_hyst[i+1,j]==255 or G_hyst[i+1,j+1]==255:
                        G_hyst[i,j] = 255
                        changed = True
                        
    return G_low, G_high, G_hyst

def hough(G_hyst):
    '''Return Hough transform of G'''
    H, W = G_hyst.shape
    D = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))
    
    num_thetas = 1000
    num_p = 2*D + 1    #(p represents rho)
    
    votes = np.zeros([num_thetas, num_p], dtype=np.int32)
    
    def ind2theta(i):
        return i / num_thetas * (np.pi)

    def p2ind(p):
        return np.int64(np.ceil(p + D))

    def ind2p(j):
        return j - D

    for row in range(H):
        for col in range(W):
            if G_hyst[row, col] > 0: 
                for i in range(num_thetas):
                    theta = ind2theta(i)
                    p = col*np.cos(theta) + row*np.sin(theta)
                    j = p2ind(p)
                    votes[i,j] += 1
                    
    return votes

def hough_low(G_hyst):
    '''Return Hough transform of G with low resolution'''
    H, W = G_hyst.shape
    D = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))
    
    num_thetas = 500  # Lower resolution for theta
    num_p = D + 1     # Lower resolution for rho
    
    votes = np.zeros([num_thetas, num_p], dtype=np.int32)
    
    def ind2theta(i):
        return i / num_thetas * (np.pi)

    def p2ind(p):
        j = np.int64(np.ceil(p + D // 2))
        j = np.clip(j, 0, num_p - 1)  # Clip j to valid range
        return j

    def ind2p(j):
        return j - D // 2

    for row in range(H):
        for col in range(W):
            if G_hyst[row, col] > 0: 
                for i in range(num_thetas):
                    theta = ind2theta(i)
                    p = col*np.cos(theta) + row*np.sin(theta)
                    j = p2ind(p)
                    votes[i,j] += 1
                    
    return votes

def hough_high(G_hyst):
    '''Return Hough transform of G with high resolution'''
    H, W = G_hyst.shape
    D = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))
    
    num_thetas = 2000  # Higher resolution for theta
    num_p = 4*D + 1    # Higher resolution for rho
    
    votes = np.zeros([num_thetas, num_p], dtype=np.int32)
    
    def ind2theta(i):
        return i / num_thetas * (np.pi)

    def p2ind(p):
        return np.int64(np.ceil(p + 2*D))

    def ind2p(j):
        return j - 2*D

    for row in range(H):
        for col in range(W):
            if G_hyst[row, col] > 0: 
                for i in range(num_thetas):
                    theta = ind2theta(i)
                    p = col*np.cos(theta) + row*np.sin(theta)
                    j = p2ind(p)
                    votes[i,j] += 1
                    
    return votes

if __name__ == '__main__':
    
    input_path = '../data/road.jpeg'
    img = read_img_as_array(input_path)
    
    # Part II: detect edges on 'img'
    gray = rgb2gray(img)
    save_path = '../data/2.1_gray.jpg' 
    save_array_as_img(gray, save_path)
    
    gray_smoothed = ndimage.gaussian_filter(gray, sigma=2)
    G, Gx, Gy = sobel(gray_smoothed) 
    save_array_as_img(Gx, '../data/2.2_G_x.jpg')
    save_array_as_img(Gy, '../data/2.2_G_y.jpg')  
    save_array_as_img(G, '../data/2.2_G.jpg')
    
    supress = nonmax_suppress(G, Gx, Gy)
    save_array_as_img(supress, '../data/2.3_supress.jpg')
    
    low_thr, high_thr = 50, 150
    edge_low, edge_high, edgemap = hysteresis_thresholding(supress, low_thr, high_thr)
    save_array_as_img(edge_low, '../data/2.4_edgemap_low.jpg')
    save_array_as_img(edge_high, '../data/2.4_edgemap_high.jpg')
    save_array_as_img(edgemap, '../data/2.4_edgemap.jpg')
    
    # Bonus: Hough transform
    hough_votes = hough(edgemap)
    save_array_as_img(hough_votes, '../data/2.5_hough.jpg') 

    flattened = hough_votes.flatten()
    sorted_indices = np.argsort(flattened)[::-1]
    top_indices = sorted_indices[:10]
    top_maximas = [(np.unravel_index(index, hough_votes.shape)) for index in top_indices]

    # Draw detected lines on image    
    from PIL import ImageDraw
    img_lines = Image.open(input_path)
    draw = ImageDraw.Draw(img_lines)  
    for i,j in top_maximas:
        theta = i / hough_votes.shape[0] * np.pi
        p = j - hough_votes.shape[1]//2

        if np.sin(theta) != 0:
            x1, x2 = 0, img_lines.width
            y1 = int((p - x1*np.cos(theta))/np.sin(theta)) 
            y2 = int((p - x2*np.cos(theta))/np.sin(theta))
        else:
            y1, y2 = 0, img_lines.height 
            x1 = x2 = int(p / np.cos(theta))

        draw.line([(x1,y1), (x2,y2)], fill=(0,0,0), width=1)

    img_lines.save('../data/2.6_detection_result.jpg')

    #Lower resolution
    hough_votes_low = hough_low(edgemap)

    flattened = hough_votes_low.flatten()
    sorted_indices = np.argsort(flattened)[::-1]
    top_indices = sorted_indices[:10]
    top_maximas = [(np.unravel_index(index, hough_votes_low.shape)) for index in top_indices]

    img_lines = Image.open(input_path)
    draw = ImageDraw.Draw(img_lines)
    for i, j in top_maximas:
        theta = i / hough_votes_low.shape[0] * np.pi
        p = j - hough_votes_low.shape[1] // 2

        if np.sin(theta) != 0:
            x1, x2 = 0, img_lines.width
            y1 = int((p - x1 * np.cos(theta)) / np.sin(theta))
            y2 = int((p - x2 * np.cos(theta)) / np.sin(theta))
        else:
            y1, y2 = 0, img_lines.height
            x1 = x2 = int(p / np.cos(theta))

        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=1)

    img_lines.save('../data/2.7_detection_result_low_resolution.jpg')

    #Higher resolution
    hough_votes_high = hough_high(edgemap)

    flattened = hough_votes_high.flatten()
    sorted_indices = np.argsort(flattened)[::-1]
    top_indices = sorted_indices[:10] 
    top_maximas = [(np.unravel_index(index, hough_votes_high.shape)) for index in top_indices]

    img_lines = Image.open(input_path)
    draw = ImageDraw.Draw(img_lines)
    for i, j in top_maximas:
        theta = i / hough_votes_high.shape[0] * np.pi
        p = j - hough_votes_high.shape[1] // 2

        if np.sin(theta) != 0:
            x1, x2 = 0, img_lines.width
            y1 = int((p - x1 * np.cos(theta)) / np.sin(theta))
            y2 = int((p - x2 * np.cos(theta)) / np.sin(theta))
        else:
            y1, y2 = 0, img_lines.height
            x1 = x2 = int(p / np.cos(theta))

        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=1)

    img_lines.save('../data/2.7_detection_result_high_resolution.jpg')