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
shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 1000 --nval 50 --ntest 50 --spc 15 --var --min_spc 5 --img_size 224 --scale_canvas_shape_ratio 0.2
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