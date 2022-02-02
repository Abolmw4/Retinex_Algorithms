import numpy as np 
import os
import cv2
from pandas import value_counts

class main:
    def __init__(self,image):
        self.img=image

    @property 
    def img(self):
        return self._img
    @img.setter
    def img(self,image):
        if image.ndim !=3:
            raise ValueError('This is not standard Image!')
        self._img=image
    def singleScaleRetinex(self,img, sigma):

        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

        return retinex
    
    def multiScaleRetinex(self,img,sigma_list):
        retinex = np.zeros_like(img)
        for sigma in sigma_list:
            retinex += self.singleScaleRetinex(img,sigma)
        retinex = retinex / len(sigma_list)
        return retinex
    
    def colorRestoration(self,img,alpha, beta):
        img_sum = np.sum(img, axis=2, keepdims=True)

        color_restoration = beta * (np.log10(alpha *img) - np.log10(img_sum))

        return color_restoration
    
    def simplestColorBalance(self,img,low_clip, high_clip):
        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):            
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
                
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

        return img

class retinex(main):
    def __init__(self, image):
        super().__init__(image)
    def MSRCR(self, sigma_list, G, b, alpha, beta, low_clip, high_clip):

        img = np.float64(self.img) + 1.0

        img_retinex = self.multiScaleRetinex(img, sigma_list)    
        img_color = self.colorRestoration(img, alpha, beta)    
        img_msrcr = G * (img_retinex * img_color + b)

        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                 (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                 255
    
        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        img_msrcr = self.simplestColorBalance(img_msrcr, low_clip, high_clip)       
        return img_msrcr 

    def automatedMSRCR(self, sigma_list):

        img = np.float64(self.img) + 1.0

        img_retinex = self.multiScaleRetinex(img, sigma_list)

        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            for u, c in zip(unique, count):
                if u == 0:
                    zero_count = c
                    break
            
            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break

            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
            img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                   (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                   * 255

        img_retinex = np.uint8(img_retinex)
        
        return img_retinex    
    
    def MSRCP(self, sigma_list, low_clip, high_clip):

        img = np.float64(self.img) + 1.0

        intensity = np.sum(img, axis=2) / img.shape[2]    

        retinex = self.multiScaleRetinex(intensity, sigma_list)

        intensity = np.expand_dims(intensity, 2)
        retinex = np.expand_dims(retinex, 2)

        intensity1 = self.simplestColorBalance(retinex, low_clip, high_clip)

        intensity1 = (intensity1 - np.min(intensity1)) / \
                     (np.max(intensity1) - np.min(intensity1)) * \
                      255.0 + 1.0

        img_msrcp = np.zeros_like(img)
    
        for y in range(img_msrcp.shape[0]):
            for x in range(img_msrcp.shape[1]):
                B = np.max(img[y, x])
                A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
                img_msrcp[y, x, 0] = A * img[y, x, 0]
                img_msrcp[y, x, 1] = A * img[y, x, 1]
                img_msrcp[y, x, 2] = A * img[y, x, 2]

        img_msrcp = np.uint8(img_msrcp - 1.0)

        return img_msrcp
