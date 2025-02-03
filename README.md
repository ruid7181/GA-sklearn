# GeoAggregator: An Efficient Transformer Model for Geo-Spatial Tabular Data

## 🌍 Introduction

This paper introduces GeoAggregator (GA), an efficient and lightweight algorithm based on the transformer architecture
designed specifically for geospatial tabular data modeling.
GAs explicitly account for spatial auto-correlation and spatial heterogeneity through Gaussian-biased local attention
and
global positional awareness.
This paper also introduces a new attention mechanism that uses the Cartesian product to manage the size of the model
while maintaining strong expressive power.

### TL;DR: In the paper, we focus on the spatial regression task...

![Research question](figs/figure_1_research_question_camera-ready.png "Workflow of the geospatial regression problem")

### ... and propose an efficient and light-weight transformer model named *GeoAggregator*.

![Architecture of GeoAggregator model](figs/figure_2_model_architecture_camera-ready.png "GeoAggregator Model Architecture")

## 🤖 Sklearn-style interface

* We provide a simple sklearn-style interface of the GeoAggregator (GA) model, to apply the model on *your own
  geospatial tabular datasets*.
  Further hyperparameter tuning is made possible through this interface.

## 🗿 Toy datasets

* [Synthetic datasets (SNR control)](data/tabular_datasets/snr-control)
* [Synthetic datasets (without SNR control)](data/tabular_datasets)]
* [3 Real-world datasets](data/tabular_datasets)

🚧 Working...

## 🔥 Usage

* The tutorial for a quick start can be found [here](demo.ipynb).