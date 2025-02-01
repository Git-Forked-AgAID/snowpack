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
- [Presentation](#presentation)
- [Preview](#preview)


## Introduction
- The problem we solved was developing and implementing a model that predicts **Snow Water Equivalent** across the Western United States.

## Pre-processing

## Training
Due to the time-series forcasting requrements, we have elected to use a Long Short-Term Memory (LSTM) Reccurent Neural Network (RNN). LSTMs are commonly used in tasks that require sequential data and squential prediction tasks. 

With our data being pre-processed, we configured the dataset and dataloaders to feed our model. We utilize an approach with a zero-gradient optimizer and the standard mean squared loss (MSE) function for our loss function. This ensures consistent and predictable training while avoiding gradient bleed.

For our hyperparameters, we are utilizing __ Epochs to ensure that enough training is completed for accurate results. We are also using a batch size of __ to ensure that we balance computing power, generalization (ability to adapt to unknown info), and accuracy. 

With all these factors combined, we are able to make accurate predications based on historical SNOWTEL data for future SWE.

## Preview
