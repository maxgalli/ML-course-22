IN main.py,

From line 78 to 81, the user must adjust the batch sizes for each set depending on whether the
code is to be run on a local machine or colab. The preset values of 120 worked flawlessly on Google Colab.

In line 82, the 'path' variable must be adjusted by the user as to reference the path of the food folder containing the images.

The working directory must contain the 'test_triplets.txt' and 'train_triplets.txt' files.


```
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/gallim/anaconda3/lib
```
