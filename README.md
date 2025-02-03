# Digital AgATH0N 2025 Snowpack prediction
**Team Name: Git Forked**

## Team Members:
- Adam Caudle
- Sean Hodgson
- Emily West
- Shane Ganz
- David Tran
- Cole Wilson

## Table of Contents
- [Introduction](#introduction)
- [Pre-Processing](#pre-processing)
- [Training](#training)
- [Model](#model)
- [Map](#map)
- [Haiku](#haiku)

## Introduction
- The problem we solved was developing and implementing a model that predicts **Snow Water Equivalent** across the Western United States.
- This was a spatiotemporal challenge, with more info in the [challenge PDF](./SnowpackPredictionChallenge.pdf).

## Products
Final Presentation Video:
[![final presentation video](https://img.youtube.com/vi/UXVqL-Rfepg/0.jpg)](https://www.youtube.com/watch?v=UXVqL-Rfepg)

Final PowerPoint Presentation: [Final_Presentation.pdf](./Final_Presentation.pdf)


## Pre-processing
Preprocessing of data requires combining several discrete datasets into one. There are several Python scripts that use GeoPandas to perform spatial joins over the datasets.
Additionally, we use mean imputation to backfill any null values, and then use scikit-learn to normalize all data into 0-1 normalized units.

The preprocessing step can be run using `./joindata.sh`. There are similar scripts such as `./combineTest.py` for doing the same to the prediction datasets.

The data used in this hackathon is private information and therefore is not in this repository.

## Training
Due to the time-series forcasting requrements, we have elected to use a **Long Short-Term Memory** (LSTM) **Reccurent Neural Network** (RNN). LSTMs are commonly used in tasks that require sequential data and squential prediction tasks.

With our data being pre-processed, we configured the dataset and dataloaders to feed our model. We utilize an approach with a zero-gradient optimizer and the standard mean squared loss (MSE) function for our loss function. This ensures consistent and predictable training while avoiding gradient bleed.

With all these factors combined, we are able to make accurate predications based on historical SNOTEL data for future SWE.

## Model
The model is made in PyTorch. There are two files `dataset.py` and `model.py` that implement the LSTM Neural Network described above. Essentially, the model will take an input dataset
and classify it using the location lat/lon into the nearest SNOTEL location. It will then use the corresponding model to that location to predict the SWE for the given time series.

## Map
Our end product goal was a map showing all of the SNOTEL sites and their photos (open source information), as well as a heatmap of SWE data across the entire Western United States.
Because the data was from discrete SNOTEL locations, we would make a gradient of continuous SWE data using our model. This map used MapBox, qGIS and more and is stored in `site/`.

## Haiku
```Snow melts far too fast,
mountains weep in silent streams,
thirsty earth awaits.```
