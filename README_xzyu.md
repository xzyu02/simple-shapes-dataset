## Install

First install the package in editable mode:
```bash
pip install -e .
```

## Usage Examples
```bash
# Single shape per image
shapesd create --output_path /users/xyu110/scratch/single --ntrain 100 --nval 20 --ntest 20 --img_size 64 --bg black

# Multiple shapes per image (fixed number)
shapesd create-multi --output_path /users/xyu110/scratch/multi --ntrain 10 --nval 5 --ntest 5 --spc 10 --img_size 224

# Variable number of shapes per image (5-10 shapes randomly)
shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 1000 --nval 50 --ntest 50 --spc 5 --var --min_spc 3 --img_size 224

# Generate QA with evenly distributed sizes for multi-shape scenes
shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 50 --nval 50 --ntest 50 --spc 3 --img_size 224 --even_sizes --captions --qa --num_qa_pairs 8

shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 50 --nval 50 --ntest 50 --spc 3 --img_size 224 --even_sizes --captions --qa --num_qa_pairs 8 --shapes "0,1,2"

# Restrict to two shapes and two colors with counting QA: CountingQADataset
shapesd create-multi \
	--output_path /users/xyu110/scratch/two-types \
	--ntrain 100 --nval 20 --ntest 20 \
	--spc 70 --img_size 224 \
	--shapes "square,circle" \
	--colors "red,green" \
	--captions --qa --qa_type counting
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
- `--shapes` restrict allowed shapes on the canvas (ids 0–6 or names like `circle,square`)
- `--colors` restrict allowed colors (comma-separated labels from COLORS_LARGE_SET, e.g. `red,green`)
- `--qa_type` type of QA to generate: `binding` (yes/no), `counting` (numeric), or `both`


## Usage Examples
```bash
# Single shape per image
shapesd create --output_path /users/xyu110/scratch --ntrain 100 --nval 20 --ntest 20 --img_size 64 --bg black

# Multiple shapes per image
shapesd create-multi --output_path /users/xyu110/scratch/multi --ntrain 10 --nval 5 --ntest 5 --spc 10 --img_size 224
```

## Create dataset
```
shapesd create --output_path "/path/to/dataset"
```
Configuration values:
- `--output_path, -o OUTPUT_PATH` where to save the dataset
- `--seed, -s SEED` random seed (defaults to 0)
- `--img_size` size of the images (defaults to 32)
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