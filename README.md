# Transfer-learning-using-VGG16
# **Classification of COVID-19 with cough COVID-19 no cough sounds**

**Problem: Classification of cough sounds into two classes:**

First-  cough sounds from COVID-19 positive users who have cough as a symptom, and

Second- cough sounds from COVID-19 positive users who do not have cough as a symptom.

**Challenge:** Both classes are close to each other as all of the users in both classes have tested positive for COVID-19, the only difference is the cough is a symptom in one class, and there is no cough as a symptom for another class.

**Solution:**

We have a limited data set - hence, we chose to implement a transfer learning approach using VGG16. VGG stands for visual geometry group, it is also called as OxfordNet.

VGG16 is a pre-trained CNN model having 16 convolutional layers and trained on millions of images from 1000 classes. The architecture of CNN is very simple as explained below:


Key points of VGG16 architecture:

1. It has 16 convolutional layers.
2. It always uses 3 x 3 kernel for convolution.
3. Max pool is 2 x 2
4. Trained on ImageNet dataset
5. The number of nodes in the output layer must be equal to the number of classes.
6. Another version is VGG19: it has 19 convolutional layers.
7. It takes input sizes - 224, 224, 3 for images.

# **Practical Approach/ Code in Python:**

The steps I followed to implement VGG16 on my dataset are explained below:

1. Preparation of dataset directories
2. Import necessary libraries
3. Import the model and load the pre-defined weights. In VGG16 the weights are pre-defined as ‘imagenet’
4. Generate the spectrograms of the dataset
5.  Divide the spectrogram images into training_set, test_set, and Validation set as explained in step 1.
6. After creating the dataset in various folders, now we will generate paths to these directories. Then see the number of spectrogram images in all folders
7. Define the size of the input image and batch_size. The input_size is always 224, 224 for VGG16. 
8. Reshape the images as per VGG16 specifications, for this we have used the module “ImageDataGenerator” from the image preprocessing of Keras.
9. Now we need to freeze the training layers of VGG16 because it is already trained.
10. As now all the layers of VGG16 are frozen, we need to modify the last layers as per our data set structure. The last layer of the VGG16 is ‘block5_pool’. I have added an extra pooling layer, dense layer, dropout layer, and an output layer to the existing architecture of VGG16. The output layer is a dense layer with 2 nodes (as it is a binary classification problem), and sigmoid activation. If it would have been a multi-class problem, then I would have used softmax activation.
11. Now we need to merge the original VGG layers with our customized layers.
12. Before training our model on the dataset, we will compile it.
I have used ‘Adam’ as an optimizer, sparse categorical cross-entropy as loss function, and accuracy rate as the metric. If it would have been a multi-class problem, then I would have chosen categorical cross-entropy as the loss function.

13.  Now, train our model for 50 epochs.
14. Next step is to test our model on the test set.
