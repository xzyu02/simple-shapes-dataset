## Installation
First install the package in editable mode:
```bash
pip install -e .
```

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
- `--bg` background color (defaults to black)
- `--ntrain` number of train examples (defaults to 500,000)
- `--nval` number of validation examples (defaults to 1000)
- `--ntest` number of test examples (defaults to 1000)
- `--bert_path, -b` which pretrained BERT model to use (defaults to "bert-base-uncased")
- `--domain_alignment, --da, -a` domain alignment to generate (see next section).

For more configurations, see
```
shapesd create --help
```