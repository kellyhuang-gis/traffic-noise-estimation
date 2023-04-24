# traffic-noise-estimation
we propose a computational framework to estimate traffic noise using street view images and deep learning approaches based on real-world measurement. The proposed model can achieve end-to-end output from street view images to road traffic noise (sound perception in decibel levels)  
## Eequirements
The python version used was **Python 3.9.7**, The requirements to execute the code is in the file **requirements.txt**
## Files Description
(1)**CDBSV**  
CDBSV contains some Street View Images of Chengdu city here used for demonstration  
    ```imgs_root = "./CDBSV"  #load street imagery that needs noise prediction```  
(2)**noise_estimation.py**  
an end-to-end model for road traffic noise estimation, the input are street view images and the output are sound perception in decibel levels
