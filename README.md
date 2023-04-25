# traffic-noise-estimation
Code associate to the manuscript "Estimating Urban Traffic Noise Levels from Street View Imagery"  
we propose a computational framework to estimate traffic noise using street view images(SVIs) and deep learning approaches based on real-world measurement. The proposed model can achieve end-to-end output from SVIs to road traffic noise (sound perception in decibel levels)  
## Eequirements
The python version used was **Python 3.9.7**, The requirements to execute the code is in the file **requirements.txt**
## Files Description  
The core of the program lies in the two python scripts:  
**(1)noise_estimation.py**  
an end-to-end program for traffic noise estimation, the input are street view images and the output are sound perception in decibel levels  
**(2)cnn_model.py**  
stores the architecture of the cnn model
    ```from cnn_model import resnet34```   
other files:  
**CDBSV**   
contains some Street View Images of Chengdu city obtained through Baidu Map API, which were utilized to showcase the functionality of the code   
    ```imgs_root = "./CDBSV"  #load street imagery that requires noise prediction```   
**class_indices.json**  
The category label of the cnn model, in our work, we used the classification-then-regression strategy to obtained traffic noise value  
    ```json_path = './class_indices.json'  #read class_indict```  
**cnn_best.pth**   
a convolutional neural network(Resnet) was trained based on the PyTorch deep learning framework to learn the noise patterns of the road environment and the .pth file is the weight of our trained Resnet network
    ```weights_path = "./cnn_best.pth"  #Load cnn weights```  
**NoiseModel.m**  
the regression algorithms were applied to compute the estimated noise values  
    ```RF = joblib.load("NoiseModel.m")  # traffic noise value estimation```  
