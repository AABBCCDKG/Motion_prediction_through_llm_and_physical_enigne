# Video_prediction_through_physical_laws

## Project Overview

Predicting 2D video sequences is a fundamental task for understanding real-world dynamics. However, existing models often require large datasets for training and struggle to predict complex motions, particularly when data is scarce or expensive. We introduce a framework for predicting object motion based on physical laws. Our approach begins by identifying the position information of objects and fitting position functions to determine the necessary dynamic parameters. These parameters are then used in a physics engine to simulate the motion of the objects. Subsequently, neural networks are employed for texture mapping, resulting in the final predicted image. By leveraging known physical laws, our framework reduces the reliance on large datasets and enhances the model’s ability to predict complex and abrupt motions.

## How to use
1. In the `main.py` file, locate the following line and fill in the number to change the input:
    ```python
    #folder_path = '/Users/dong/Desktop/video/dataset/{1-10}/{1-2}/input/'
{1-10}: 10 video groups, each containing videos with interaction and without interaction 

{1-2}: 1-with interaction, 2-without interaction

2. Run the `main.py` file.

## Future Work
Complete the training of neural network based on cGAN for texture mapping
