# traffic-noise-estimation
Code associate to the manuscript "Estimating urban noise along road network from street view imagery."  
A computational framework to estimate traffic noise using street view images (SVIs) and deep learning approaches based on real-world measurement. The proposed model can achieve end-to-end output from SVIs to road traffic noise (sound perception in decibel levels)  
## Eequirements
The python version used was **Python 3.9.7**, The requirements to execute the code is in the file **requirements.txt**
## Files Description  
### The core of the program lies in the two python scripts:  
**(1)noise_estimation.py**  
an end-to-end program for traffic noise estimation, the input are street view images and the output are sound perception in decibel levels  
**(2)cnn_model.py**  
stores the architecture of the cnn model
    ```from cnn_model import resnet34```   
### Other files:  
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
**result.xlsx**  
The ultimate outcome encompasses the output probability vector of the CNN, the estimated traffic noise value(quantified in decibels), as well as the identifier for the corresponding street view image  
## Frequently Asked Questions (FAQ):   
Q: What will happen/how will the algorithm work if the vector (𝑞0,𝑞1,𝑞2,𝑞3,𝑞4) equals (0.3, 0.15, 0.1, 0.15, 0.3) or (0.1, 0.15, 0.15, 0.3, 0.3)? It means that there can be two or more items having the same dominant probability.   
A: In theory, it is rare for the probability vector of a convolutional neural network (CNN) classification output to have identical values for two vector classes. CNN employs backpropagation algorithm during training to optimize parameters and minimize the loss function, resulting in an accurate representation of each category's probability on the training data. Nevertheless, practical considerations, such as limitations in computer floating-point accuracy and model complexity, can occasionally lead to small numerical errors, causing closely related, but not identical, values in probability vectors. Additionally, in certain scenarios, multiple classes in the probability vector may have the same probability value.  Such cases often arise when the model exhibits significant uncertainty in classifying certain image samples or when the dataset itself suffers from class imbalance issues. However, it is crucial to note that the likelihood of encountering such probabilities is exceedingly low.
To provide empirical evidence, we conducted a comprehensive data analysis on 10,000 instances. Our findings reveal that the difference between the dominant probability and the sub-dominant probability is less than 1% in merely 0.34% of cases, affirming the rarity of these occurrences in the dataset.  
**outputprob_analysis.py**  
This code reads data from an Excel spreadsheet (outputprob_portion.xlsx) and randomly selects 10,000 rows from it. For each of these 10,000 rows, it identifies the column with the maximum value and the column with the second-highest value. The code then calculates the difference between the values in these two columns and counts the number of differences whose absolute value is less than 0.01.   
**outputprob_portion.xlsx**    
output probability of CNN.
