#Blur Detection

There are many methods to identify blurred images. Some of these are:
1. Compute **Fast Fourier Transform (FFT)** of the image and then examine the distribution of low and high frequencies. If there are low amount of high frequencies, then the image can be considered blurry. The challenge is determining the threshold for the number of high frequencies.
2. **Variation of Laplacian:** Convolve a single channel image with 3*3 Laplacian kernel and find the variance of the output. If the variance is below a threshold, then the image is considered blurry. Again, the threshold needs to be manually determined by examining the variance of the blurred images
3. **Deep Learning:** Train a convolutional neural network (CNN) on images dataset to classify images as blurred/non-blurred. The CNN can be trained on any unlabelled dataset similar to the images present in testset. If the testset images are of human faces, then the CNN can be trained on Flickr-Faces-HQ (FFHQ) dataset (https://github.com/NVlabs/ffhq-dataset).

In this code, I implement the last two approaches. The CNN is trained on 3000 images from FFHQ dataset where half the images are blurred with Gaussian blur having radius between 3 and 9. The blurring is perfomed on only 50% of the pixels in the image because in the test images some examples have blurring in only certain regions of the image (moving mouth). The network is trained to perform binary classification. The trained network is tested on the custom dataset of human faces. After evaluation, the blurred and clean images are stored in different folders

#Improvements
The deep learining model can be improved by trying different kind of blurring effects during training. The Also, the model can be improved by choosing a deeper VGG or ResNet architecture.