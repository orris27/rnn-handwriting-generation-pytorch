## Generating handwriting with LSTM, Mixture Gaussian & Bernoulli distribution and PyTorch 

This is a PyTorch implementation of *[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)* by Alex Graves.

This code is based on *[rnn-handwriting-generation](https://github.com/snowkylin/rnn-handwriting-generation)*. Many thanks for the authors!

This repo only has one function:

* **Handwriting Prediction**: Randomly generate a line of handwriting (set `mode=predict`). 

### Sample Result

#### Handwriting Prediction

![sample.normal.svg](./sample.normal.svg)
This is the result with default setting:
* rnn state size = 256
* rnn length = 300
* num of layers = 2
* number of mixture gaussian = 20
* epochs: 20+
