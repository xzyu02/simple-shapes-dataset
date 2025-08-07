## Install

First install the package in editable mode:
```bash
pip install -e .
```

## Usage Examples
```bash
# Single shape per image
shapesd create --output_path /users/xyu110/scratch --ntrain 100 --nval 20 --ntest 20 --img_size 64 --scale_canvas_shape_ratio 1.0 --bg black

# Multiple shapes per image (fixed number)
shapesd create-multi --output_path /users/xyu110/scratch/multi --ntrain 10 --nval 5 --ntest 5 --spc 10 --img_size 224 --scale_canvas_shape_ratio 0.2

# Variable number of shapes per image (5-10 shapes randomly)
shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 1000 --nval 50 --ntest 50 --spc 5 --var --min_spc 3 --img_size 224 --auto_size_range --size_distribution balanced

# Caption Generation for Multi Shapes with optimal sizing
shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 50 --nval 50 --ntest 50 --spc 5 --var --min_spc 3 --img_size 224 --even_sizes --captions

# Generate QA with evenly distributed sizes for multi-shape scenes
shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 50 --nval 50 --ntest 50 --spc 7 --var --min_spc 5 --img_size 224 --even_sizes --captions --qa --num_qa_pairs 8
```

Results running `test_dataset_loader.py`:
```bash
[xyu110@gpu2261 simple-shapes-dataset]$ python test_dataset_loader.py 

================================================================================
DATASET SUMMARY: /users/xyu110/scratch/variable
================================================================================

--- TRAIN SPLIT ---
✓ Loaded captions for train
✓ Loaded caption_choices for train
✓ Loaded latent for train
✓ Loaded unpaired for train
Number of canvases: 96
captions: (1000,) items
caption_choices: (1000,) items
latent: (1000, 768) items
unpaired: (672, 15, 32) items
Image files: 1000 PNG files

--- VAL SPLIT ---
✓ Loaded captions for val
✓ Loaded caption_choices for val
✓ Loaded latent for val
✓ Loaded unpaired for val
Number of canvases: 5
captions: (50,) items
caption_choices: (50,) items
latent: (50, 768) items
unpaired: (35, 15, 32) items
Image files: 50 PNG files

--- TEST SPLIT ---
✓ Loaded captions for test
✓ Loaded caption_choices for test
✓ Loaded latent for test
✓ Loaded unpaired for test
Number of canvases: 5
captions: (50,) items
caption_choices: (50,) items
latent: (50, 768) items
unpaired: (35, 15, 32) items
Image files: 50 PNG files

================================================================================
DATASET INSPECTION: /users/xyu110/scratch/variable
Split: train, Image Index: 0
================================================================================

--- DATASET METADATA ---
shapes_per_canvas: 15
variable_shapes: True
min_shapes_per_canvas: 5
max_shapes_per_canvas: 15
img_size: 224
min_scale: 7
max_scale: 14
actual_min_scale: 9
actual_max_scale: 19
background_color: black
version: 1.1.0
✓ Loaded captions for train
✓ Loaded caption_choices for train
✓ Loaded latent for train
✓ Loaded unpaired for train

--- IMAGE INFO ---
Image path: /users/xyu110/scratch/variable/train/0.png
Image exists: True

--- SHAPE ATTRIBUTES ---
Number of shapes: 10

Shape 1 (shape_idx=0):
  Type: diamond (class 0)
  Location: (29, 91) (x, y)
  Size: 20 pixels (8.9% of canvas)
  Rotation: 2.468 radians
  Color RGB: (61, 60, 37)
  Color HLS: (29, 49, 62)
  Unpaired attr: 0.945

Shape 2 (shape_idx=1):
  Type: circle (class 3)
  Location: (144, 117) (x, y)
  Size: 20 pixels (8.9% of canvas)
  Rotation: 5.253 radians
  Color RGB: (187, 146, 240)
  Color HLS: (133, 193, 194)
  Unpaired attr: 0.522

Shape 3 (shape_idx=2):
  Type: circle (class 3)
  Location: (56, 200) (x, y)
  Size: 24 pixels (10.7% of canvas)
  Rotation: 2.120 radians
  Color RGB: (95, 12, 104)
  Color HLS: (147, 58, 203)
  Unpaired attr: 0.415

Shape 4 (shape_idx=3):
  Type: circle (class 3)
  Location: (44, 133) (x, y)
  Size: 22 pixels (9.8% of canvas)
  Rotation: 4.073 radians
  Color RGB: (204, 172, 180)
  Color HLS: (173, 188, 59)
  Unpaired attr: 0.265

Shape 5 (shape_idx=4):
  Type: oval (class 1)
  Location: (8, 72) (x, y)
  Size: 13 pixels (5.8% of canvas)
  Rotation: 2.314 radians
  Color RGB: (26, 126, 63)
  Color HLS: (71, 76, 167)
  Unpaired attr: 0.774

Shape 6 (shape_idx=5):
  Type: circle (class 3)
  Location: (179, 67) (x, y)
  Size: 18 pixels (8.0% of canvas)
  Rotation: 6.014 radians
  Color RGB: (107, 202, 231)
  Color HLS: (97, 169, 184)
  Unpaired attr: 0.456

Shape 7 (shape_idx=6):
  Type: star (class 5)
  Location: (130, 22) (x, y)
  Size: 19 pixels (8.5% of canvas)
  Rotation: 0.882 radians
  Color RGB: (255, 255, 255)
  Color HLS: (32, 255, 163)
  Unpaired attr: 0.568

Shape 8 (shape_idx=7):
  Type: triangle (class 2)
  Location: (185, 93) (x, y)
  Size: 19 pixels (8.5% of canvas)
  Rotation: 5.467 radians
  Color RGB: (192, 196, 192)
  Color HLS: (63, 194, 9)
  Unpaired attr: 0.019

Shape 9 (shape_idx=8):
  Type: square (class 4)
  Location: (106, 143) (x, y)
  Size: 26 pixels (11.6% of canvas)
  Rotation: 2.976 radians
  Color RGB: (141, 23, 30)
  Color HLS: (178, 82, 185)
  Unpaired attr: 0.618

Shape 10 (shape_idx=9):
  Type: heart (class 6)
  Location: (133, 182) (x, y)
  Size: 20 pixels (8.9% of canvas)
  Rotation: 5.032 radians
  Color RGB: (157, 225, 189)
  Color HLS: (74, 191, 137)
  Unpaired attr: 0.612

--- FOL CAPTION ---
Caption: a very small dark slate grey diamond, a plum round shape, a tiny, indigo round shape, a very small, silver round shape, a forest green guitar pick, a skyblue circular object, a white star, a silver isosceles triangle, a tiny, brown four-sided rectangle, and a tiny light blue heart shape are visible.
Caption choices/metadata:
  shape_0: {'name': 0}
  size_0: tiny
  color_0: {'val': 0}
  shape_1: {'name': 1}
  size_1: tiny
  color_1: {'val': 0}
  shape_2: {'name': 1}
  size_2: tiny
  color_2: {'val': 0}
  shape_3: {'name': 1}
  size_3: tiny
  color_3: {'val': 0}
  shape_4: {'name': 6}
  size_4: tiny
  color_4: {'val': 0}
  shape_5: {'name': 2}
  size_5: tiny
  color_5: {'val': 0}
  shape_6: {'name': 0}
  size_6: tiny
  color_6: {'val': 0}
  shape_7: {'name': 0}
  size_7: tiny
  color_7: {'val': 0}
  shape_8: {'name': 2}
  size_8: tiny
  color_8: {'val': 0}
  shape_9: {'name': 1}
  size_9: tiny
  color_9: {'val': 0}
  template: simple_conjunction

--- BERT EMBEDDINGS ---
Embedding shape: (768,)
Embedding stats: mean=-0.0095, std=0.5271
First 10 values: [-0.5246011  -0.03532032 -0.6665657   0.14821616 -0.09840114  0.02904496
  0.02671585  0.96085364 -0.3952752  -0.20731704]

--- UNPAIRED ATTRIBUTES ---
Unpaired shape: (15, 32)
Unpaired stats: mean=-0.0312, std=0.9914

================================================================================

================================================================================
DATASET INSPECTION: /users/xyu110/scratch/variable
Split: train, Image Index: 1
================================================================================

--- DATASET METADATA ---
shapes_per_canvas: 15
variable_shapes: True
min_shapes_per_canvas: 5
max_shapes_per_canvas: 15
img_size: 224
min_scale: 7
max_scale: 14
actual_min_scale: 9
actual_max_scale: 19
background_color: black
version: 1.1.0
✓ Loaded captions for train
✓ Loaded caption_choices for train
✓ Loaded latent for train
✓ Loaded unpaired for train

--- IMAGE INFO ---
Image path: /users/xyu110/scratch/variable/train/1.png
Image exists: True

--- SHAPE ATTRIBUTES ---
Number of shapes: 9

Shape 1 (shape_idx=0):
  Type: circle (class 3)
  Location: (160, 62) (x, y)
  Size: 20 pixels (8.9% of canvas)
  Rotation: 1.993 radians
  Color RGB: (238, 245, 227)
  Color HLS: (42, 236, 118)
  Unpaired attr: 0.442

Shape 2 (shape_idx=1):
  Type: square (class 4)
  Location: (170, 29) (x, y)
  Size: 26 pixels (11.6% of canvas)
  Rotation: 4.890 radians
  Color RGB: (251, 251, 251)
  Color HLS: (130, 251, 3)
  Unpaired attr: 0.980

Shape 3 (shape_idx=2):
  Type: heart (class 6)
  Location: (50, 67) (x, y)
  Size: 16 pixels (7.1% of canvas)
  Rotation: 5.966 radians
  Color RGB: (189, 154, 127)
  Color HLS: (13, 158, 81)
  Unpaired attr: 0.359

Shape 4 (shape_idx=3):
  Type: square (class 4)
  Location: (202, 45) (x, y)
  Size: 15 pixels (6.7% of canvas)
  Rotation: 4.163 radians
  Color RGB: (6, 114, 31)
  Color HLS: (67, 60, 231)
  Unpaired attr: 0.481

Shape 5 (shape_idx=4):
  Type: square (class 4)
  Location: (23, 99) (x, y)
  Size: 22 pixels (9.8% of canvas)
  Rotation: 0.085 radians
  Color RGB: (213, 232, 168)
  Color HLS: (39, 200, 149)
  Unpaired attr: 0.689

Shape 6 (shape_idx=5):
  Type: circle (class 3)
  Location: (54, 115) (x, y)
  Size: 19 pixels (8.5% of canvas)
  Rotation: 3.913 radians
  Color RGB: (178, 210, 136)
  Color HLS: (43, 173, 115)
  Unpaired attr: 0.880

Shape 7 (shape_idx=6):
  Type: square (class 4)
  Location: (26, 17) (x, y)
  Size: 25 pixels (11.2% of canvas)
  Rotation: 4.233 radians
  Color RGB: (28, 159, 192)
  Color HLS: (96, 110, 189)
  Unpaired attr: 0.918

Shape 8 (shape_idx=7):
  Type: square (class 4)
  Location: (61, 90) (x, y)
  Size: 17 pixels (7.6% of canvas)
  Rotation: 6.107 radians
  Color RGB: (74, 74, 76)
  Color HLS: (127, 75, 5)
  Unpaired attr: 0.217

Shape 9 (shape_idx=8):
  Type: square (class 4)
  Location: (42, 192) (x, y)
  Size: 17 pixels (7.6% of canvas)
  Rotation: 5.518 radians
  Color RGB: (103, 97, 97)
  Color HLS: (0, 100, 8)
  Unpaired attr: 0.565

--- FOL CAPTION ---
Caption: a very small beige circle, a tiny snow four-sided rectangle, a very small rosy brown heart, a tiny dark green four-sided rectangle, a tiny pale goldenrod four-sided rectangle, a tiny dark khaki circular object, a very small light sea green square, a very small, dark slate grey square, and a very small, dim grey square are visible.
Caption choices/metadata:
  shape_0: {'name': 0}
  size_0: tiny
  color_0: {'val': 0}
  shape_1: {'name': 2}
  size_1: tiny
  color_1: {'val': 0}
  shape_2: {'name': 0}
  size_2: tiny
  color_2: {'val': 0}
  shape_3: {'name': 2}
  size_3: tiny
  color_3: {'val': 0}
  shape_4: {'name': 2}
  size_4: tiny
  color_4: {'val': 0}
  shape_5: {'name': 2}
  size_5: tiny
  color_5: {'val': 0}
  shape_6: {'name': 0}
  size_6: tiny
  color_6: {'val': 0}
  shape_7: {'name': 0}
  size_7: tiny
  color_7: {'val': 0}
  shape_8: {'name': 0}
  size_8: tiny
  color_8: {'val': 0}
  template: simple_conjunction

--- BERT EMBEDDINGS ---
Embedding shape: (768,)
Embedding stats: mean=-0.0106, std=0.5510
First 10 values: [-0.48592237 -0.00794679 -0.6760182  -0.01162283 -0.24384476  0.24309006
 -0.16770256  1.1587838  -0.6497931  -0.4137972 ]

--- UNPAIRED ATTRIBUTES ---
Unpaired shape: (15, 32)
Unpaired stats: mean=-0.0777, std=0.9990

================================================================================
```

## Create Multi-Shapes Dataset
```
shapesd create-multi --output_path "/path/to/dataset"
```
Multi-shapes specific options:
- `--spc, --shapes_per_canvas` number of shapes per canvas (fixed mode) or max shapes (variable mode)
- `--var, --variable_shapes` enable variable number of shapes per canvas
- `--min_spc, --min_shapes_per_canvas` minimum shapes per canvas when using variable mode (default 1)
- `--captions` generate text captions (experimental)


## Usage Examples
```bash
# Single shape per image
shapesd create --output_path /users/xyu110/scratch --ntrain 100 --nval 20 --ntest 20 --img_size 64 --scale_canvas_shape_ratio 1.0 --bg black

# Multiple shapes per image
shapesd create-multi --output_path /users/xyu110/scratch/multi --ntrain 10 --nval 5 --ntest 5 --spc 10 --img_size 224 --scale_canvas_shape_ratio 0.3
```

## Create dataset
```
shapesd create --output_path "/path/to/dataset"
```
Configuration values:
- `--output_path, -o OUTPUT_PATH` where to save the dataset
- `--seed, -s SEED` random seed (defaults to 0)
- `--img_size` size of the images (defaults to 32)
- `--scale_canvas_shape_ratio` shape ratio to canvas (defaults to 0.0), suggesting 0.0 to 1.0, 1.0 keeps original shape size and smaller number downscale it
- `--bg` background color (defaults to black), supports "black", "blue", "gray", "noise".
- `--ntrain` number of train examples (defaults to 500,000)
- `--nval` number of validation examples (defaults to 1000)
- `--ntest` number of test examples (defaults to 1000)
- `--bert_path, -b` which pretrained BERT model to use (defaults to "bert-base-uncased")
- `--domain_alignment, --da, -a` domain alignment to generate (see next section).

For more configurations, see
```
shapesd create --help
```