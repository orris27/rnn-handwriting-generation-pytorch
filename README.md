## Generating handwriting with LSTM, Mixture Gaussian & Bernoulli distribution and PyTorch 

This is a PyTorch implementation of *[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)* by Alex Graves.

This code is based on *[rnn-handwriting-generation](https://github.com/snowkylin/rnn-handwriting-generation)*. Many thanks for the authors!

This repo only has one function:

* **Handwriting Prediction**: Randomly generate a line of handwriting (set `mode=predict`). 

### Sample Result

#### Handwriting Prediction

![sample.normal.svg](https://cdn.rawgit.com/snowkylin/rnn-handwriting-generation/master/sample.normal.svg)
This is the result with default setting:
* rnn state size = 256
* rnn length = 300
* num of layers = 2
* number of mixture gaussian = 20

and 20+ epochs. Not so fancy but can be recognized as something like handwritting, huh?

#### Handwriting Synthesis

![sample.normal.biased.3.svg](https://cdn.rawgit.com/snowkylin/rnn-handwriting-generation/master/sample.normal.biased.3.svg)

This is the result with the string "a quick brown fox jumps over the lazy dog".

In addition, [the scribe project by greydanus](https://github.com/greydanus/scribe) also helps me a lot, expecially the use of `tf.batch_matmul()`.
