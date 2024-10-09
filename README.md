# Sparse Attention Decomposition Applied to Circuit Tracing

Code to reproduce the paper ["Sparse Attention Decomposition Applied to Circuit Tracing"](https://arxiv.org/abs/2410.00340).

## Requirements

```sh
conda env create -f environment.yml
conda activate sparse-attn-decomp
```

## Reproduce paper figures

See ```paper.ipynb```.

## Tracing 

### Run the tracing and saving its data

```sh
python3 full_tracing_data_collection.py
```

This script produces the file ```results.nms-p256-f1.0-folded-expandedO-scaled.pkl```, which is used for the next experiments and plots.

### Build the graphs

```sh
python3 full_tracing_build_graph.py
```

This script produces the files ```nms-p256-f1.0-folded-expandedO-scaled.graphml```, which is used for the next experiments and plots.

## Interventions

### Single-edge

```sh
python3 interventions_single_edge.py
```

This script produces the file ```interventions_single-edge.parquet```, which is used for the plots.

### Multi-edge

```sh
python3 interventions_multi_edge.py
```

This script produces the file ```interventions_multi-edge.parquet```, which is used for the plots.


