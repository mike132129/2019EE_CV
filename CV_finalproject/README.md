Before testing our final project, please put the synthetic images and their ground truth in the folder: ./data/Synthetic, and put the real images in the folder: ./data/real .

For testing our final project, please try:
$ python3 main.py --input-left <path to left image> --input-right <path to right image> --output <path to output PFM file>

For example:
$ python3 main.py --input-left ./data/Synthetic/TL0.png --input-right ./data/Synthetic/TR0.png --output ./result1/TL0.pfm
$ python3 main.py --input-left ./data/Real/TL0.bmp --input-right ./data/Real/TR0.bmp --output ./result2/TL0.pfm

Note: The synthetic disparity will be saved in ./result1, and the real disparity will be saved in ./result2
Note: The runtime will be shown on the terminal.

To visualize our result, please try:
$ python3 visualize.py <path to disparity>

For example:
$ python3 visualize.py ./result2/TL0.bmp

To evaluate our synthetic average score, please try:
$ python3 score.py <path to ground truth> <path to our disparity>

For example
$ python3 score.py ./data/synthetic/TLD0.pfm ./result1/TL0.pfm
