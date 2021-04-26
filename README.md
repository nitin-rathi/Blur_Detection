# Blur Detection

There are many methods to identify blurred images. Some of these are:
1. Compute **Fast Fourier Transform (FFT)** of the image and then examine the distribution of low and high frequencies. If there are low amount of high frequencies, then the image can be considered blurry. The challenge is determining the threshold for the number of high frequencies.
2. **Variation of Laplacian:** Convolve a single channel image with 3*3 Laplacian kernel and find the variance of the output. If the variance is below a threshold, then the image is considered blurry. Again, the threshold needs to be manually determined by examining the variance of the blurred images
3. **Deep Learning:** Train a convolutional neural network (CNN) on images dataset to classify images as blurred/non-blurred. The CNN can be trained on any unlabelled dataset similar to the images present in testset. If the testset images are of human faces, then the CNN can be trained on Flickr-Faces-HQ (FFHQ) dataset (https://github.com/NVlabs/ffhq-dataset).

In this code, I implement the last two approaches. The CNN is trained on 3000 images from FFHQ dataset where half the images are blurred with Gaussian/Box blur having radius between 3 and 6. The network is trained to perform binary classification. The trained network is tested on the custom dataset of human faces. After evaluation, the blurred and clean images are stored in different folders

# Improvements
- The deep learining model can be improved by employing datasets that provide face location and blurring only the face and not the background. Currently, many of the images that are classified as blurred have sharp face but blurry background due to out of focus.
- Model can be improved by choosing a deeper VGG or ResNet architecture.
