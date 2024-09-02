# MLX Flux experiments
Some code I used to experiment with Flux-schnell on an M2 macbook pro using MLX. 

## generate.py
Allows two types of image generation: prompt driven generation and iterative image driven generation. 
As the name suggests, prompt driven generation simply generates an image from a given prompt. Iterative image driven generation on the other hand, iteratively generates an image based on a previously generated image.

Prompt driven generation example:
```
python generate.py prompt "A snow monkey sitting in a hotspring, photorealistic" monke.png --steps 100 --seed 123
```
Iterative image driven generation example:
```
python generate.py image_loop "monke.png" "monke" 900 --steps 10 --denoise 0.23 --seed 123 --cont
```
Here, the script will iteratively generate 900 images in the directory 'monke'. `--cont` tells the script to continue from the last image in the directory 'monke', if this exists. 

## createVideo.py
This script converts the iteratively generated images from the image_loop mode into a video.
```
python createVideo.py monke 16 monke.mp4
```
In the example above, a framerate of 16 has been chosen.
