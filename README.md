# Julia Neural Network Visualization

This application is a customized version based on the guide from [FluxML](https://fluxml.ai/Flux.jl/stable/guide/models/quickstart/). It demonstrates the training and visualization of a neural network model using Julia and Flux. It also allows us to save trained model in bson format.

## Requirements

To run this application, ensure you have Julia installed. Navigate to the root folder of this project in your terminal and activate the Julia environment with `] activate .`

## Usage

### Training and Visualizing Directly

To train the model and visualize the results directly run the following command and pass the number of training steps (default = 1000):

   ```include("src/PlotFromData.jl")```

### Training and Saving the Model

To train the model and save it run the following command and pass the number of training steps (default = 1000):

```include("src/SaveModel.jl")```

### Visualizing with a Pre-trained Model

To load the model and visualize the results run the following command and pass the number of training steps(models must be saved first):

```include("src/PlotFromModel.jl")```

