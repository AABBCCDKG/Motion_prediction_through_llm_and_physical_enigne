# Motion_prediction_through_llm_and_physical_enigne

## Project Overview

Predicting 2D video sequences is a fundamental task for understanding real-world dynamics. However, existing models often require large datasets for training and struggle to predict complex motions, particularly when data is scarce or expensive. We introduce a framework for predicting object motion based on physical laws. Our approach begins by identifying the position information of objects and fitting position functions to determine the necessary dynamic parameters. These parameters are then used in a physics engine to simulate the motion of the objects. Subsequently, neural networks are employed for texture mapping, resulting in the final predicted image. This method eliminates the need for extensive datasets and effectively handles situations involving sudden changes, such as collisions.

## How to use
1. In the `main.py` file, locate the following line and fill in the number to change the input:
    ```python
    #folder_path = '/Users/dong/Desktop/video/dataset/{1-10}/{1-2}/input/'
{1-10}: 10 video groups, each containing videos with interaction and without interaction 

{1-2}: 1-with interaction, 2-without interaction

2. Run the `main.py` file.

## Future Work
Complete the training of neural network based on cGAN for texture mapping
