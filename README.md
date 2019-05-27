# Malaria-Detector-CNN
This project is a 3-layer, fully connected Convolutional Neural Network (CNN) designed to detect cell images which have been parasitized with malaria.  This is my capstone project for Udacity's Machine Learning nano-degree.

## Background
This project was inspired by a Kaggle competition project called “Malaria Cell Images Dataset - Cell Images for Detecting Malaria”. The intent is to save humans by developing an algorithm to determine whether image cells show infestation by malaria. The dataset of images (which actually comes from the NIH) contains cell images in two categories – those parasitized by malaria and those uninfected. Ref 1

Malaria is “a mosquito-borne disease caused by a parasite. People with malaria often experience fever, chills, and flu-like illness. Left untreated, they may develop severe complications and die. In 2016 an estimated 216 million cases of malaria occurred worldwide and 445,000 people died, mostly children in the African Region. About 1,700 cases of malaria are diagnosed in the United States each year. The vast majority of cases in the United States are in travelers and immigrants returning from countries where malaria transmission occurs, many from sub-Saharan Africa and South Asia.” Ref 2

## Algorithm Class :
3-layer, fully connected Convolutional Neural Network (CNN)

## Problem Type
Classification of large dataset (>27K images)

## Benchmark
A benchmark was established using a 1-layer CNN.  The model design is shown below:

    #Initialize the CNN
    benchmark_model = Sequential()

    #Add convolution layer
    benchmark_model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    #Add pooling layer
    benchmark_model.add(MaxPooling2D(pool_size = (2, 2)))

    #Add flattening layer
    benchmark_model.add(Flatten())

    #Add fully connected layer
    benchmark_model.add(Dense(units = 128, activation = 'relu'))
    benchmark_model.add(Dense(units = 2, activation = 'softmax'))

    #Compile the CNN
    benchmark_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])*

Accuracy : %

Receiver Operating Characteristics (ROC) and Confusion Matrix

## 3-layer fully connected CNN design
The following structure was estalished for the model

    #Initialize the CNN
    model = Sequential()

    #Add 1st convolutional layer
    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    #Add 2nd convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    #Add 3rd convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    #Add fully connected layer
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 2, activation = 'softmax'))

    #Compile the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Final Result :
A 3-layer CNN was developed 
Accuracy : %

Receiver Operating Characteristics (ROC) and Confusion Matrix

### 3-layer fully connected CNN

Accuracy : %

Receiver Operating Characteristics (ROC) and Confusion Matrix

![](Final_Quadcopter_Reward_Plot.png)

# Project Instructions
1. Clone the repository and navigate to the downloaded folder.

~~~~
git clone https://github.com/davidsprice/Quadcopter
cd Quadcopter
~~~~

2. Create and activate a new environment.

~~~~
conda create -n quadcopter python=3.6 matplotlib numpy pandas keras-gpu
source activate quadcopter
~~~~

3. Create an [IPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the quadcopter environment.

~~~~
python -m ipykernel install --user --name quadcop --display-name "quadcop"
~~~~

4. Open the notebook.

~~~~
jupyter notebook Quadcopter_Project.ipynb
~~~~

5. Before running code, change the kernel to match the quadcop environment by using the drop-down menu (Kernel > Change kernel > quadcop). Then, follow the instructions in the notebook.

6. You will likely need to install more pip packages to complete this project. Please curate the list of packages needed to run your project in the requirements.txt file in the repository.

# References

1	https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
2	https://www.cdc.gov/parasites/malaria/index.html
3	https://ceb.nlm.nih.gov/repositories/malaria-datasets/
4	https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria/discussion/80214
