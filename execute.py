from origin import retinex
import cv2
import matplotlib.pyplot as plt
import os 

image_path=os.path.join(os.getcwd(),'screenshot.jpg')
img=cv2.imread(image_path)
r=retinex(img)

#cv2.imshow('MSRCP',r.MSRCP([10,12,14,16],0.01,0.99))
#cv2.imshow('MSRCR',r.MSRCR([10,12,14,16],5.0,25.0,125.0,46.0,0.01,0.99))
#cv2.imshow('automatedMSRCR',r.automatedMSRCR([10,12,14,16]))
#cv2.waitKey()
#cv2.destroyAllWindows()




# code for displaying multiple images in one figure

#import libraries
# create figure
fig = plt.figure(figsize=(30,30))

# setting values to rows and column variables
rows = 2
columns = 2
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(r.MSRCP([10,12,14,16],0.01,0.99))
plt.axis('off')
plt.title("MSRCP")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(r.MSRCR([10,12,14,16],5.0,25.0,125.0,46.0,0.01,0.99))
plt.axis('off')
plt.title("MSRCR")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(r.automatedMSRCR([10,12,14,16]))
plt.axis('off')
plt.title("automatedMSRCR")
plt.show()
