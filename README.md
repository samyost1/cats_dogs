# cats_dogs
PyTorch Image Classification Model for Cat and Dog Pictures.

Using the PyTorch library a model can be trained and tested on the provided dataset. The script uses a pre-trained model from torchvision's models as a base and finetunes it on the provided dataset. The script allows for loading a trained model if one already exists, otherwise it will train a new model and save it for future use. After training or loading the model it's tested with the test data and prints the resulting accuracy. It also includes a function for testing with a single image file and displaying the results.


<h2>Example results</h2>
<div style="display:flex">
     <div style="flex:1;padding-right:10px;">
          <img src="examples/example_1.png" width="500"/>
     </div>
     <div style="flex:1;padding-left:10px;">
          <img src="examples/example_2.png" width="500"/>
     </div>
     <div style="flex:1;padding-left:10px;">
          <img src="examples/example_3.png" width="500"/>
     </div>
</div>