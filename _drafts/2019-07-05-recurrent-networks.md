---
layout: post
author: Colin Swaney
title: Recurrent Neural Networks
date: 2019-07-05
categories: [research]
category: research
tags: [deep-learning]
---
An introduction to recurrent neural networks.

# Intuition
Linguists (at least Noam Chomsky) will tell you that a critical feature of language is that it is "nonlinear". As a mathematically-oriented folk, I always find this statement confusing: what are $$y$$ and $$x$$? What is meant by this statement is that the meaning of a sentence doesn't "flow" from start-to-finish. It is often the case that a critical part to understanding the end of a sentence occurs near the beginning, which can be arbitrarily far from the end...

What all of this boils down to is that language requires a model that is capable of learning complex interactions across potentially vast periods. An recurrent neural network is designed to accomplish both.

Importantly, the probability of a particular word at any point in a sentence depends on *everything* that has come before it (and really, on everything that comes after it--but let's ignore that for now).

# Mathematical Formulation
A basic RNN is a state-space model with a "forcing" variable

$$ p(s_t | s_{t - 1}, x_{t - 1}) = g(s_{t - 1}, x_{t - 1})$$

$$ p(y_t | s_t) = f(s_t) $$


The input $$x_t$$ is taken to be a purely exogenous variable--it's just some input to the system (e.g. someone entering key-strokes). $$s_t$$ is the hidden state, a possibly real, possibly purely fictional entity that drives the system. $$s_t$$ might be observable in principle, and is simply hidden from us in the data, or it might be truly unobservable--it doesn't matter to us why we can't see it.

Two things happen each time step. First, the state is going to transition, $$ s_{t - 1} \rightarrow s_t$$. Next, the output $$y_t$$ is revealed. In a machine learning course focused on probability we would specify how these events occur by saying something like

$$ s_t \ \vert \ s_{t - 1}, x_t \sim p(s_{t - 1}, x_t \ \vert \ \theta_s) $$

$$ y_t \ \vert \ s_t \sim p(s_t \ \vert \ \theta_y) $$

Just like a "normal" neural network, we don't make assumptions about forms--we'll just assume that the distributions above can be well-approximated by some possibly complex functional form plus noise. Therefore, the model is extremely expressive, but also data-hungry. For the data scientists in the crowd, the model counts on bias-reduction dominating variance-amplification.

Here's the graph of our model:

![RNN model graph](/assets/rnn_model_graph.jpg)


# Network Representation
The idea is the same as always with neural networks: anywhere you see a probability distribution, replace it with (at least one) hidden layer. In the case of the basic RNN model, we will need a hidden layer to represent the transition probability, and another layer to represent the output probability.

$$ s_t = \tanh \left(W_{xs} x_t + W_{ss} s_{t - 1} + b_s \right) $$

$$ y_t = W_{sy} s_t + b_y $$

If the output is categorical (as is often the case), then $$ y_t = \text{softmax}\left(W_{sy} s_t + b_y\right).$$ This combination of operations is often referred to an RNN "cell". We can increase the expressiveness of the network by combining cells, in which case the output of the first cell becomes the input of the next cell, and each cell maintains its own state. Effectively, the combination amounts to increasing the dimension of the state space and deepening the network. And it is of course possible to tack on non-recurrent layers to the combination (or to a single cell). For example, you might see an RNN cell followed by a single fully-connected layer, similar to the architecture of a typical convolutional network.

It's worth pointing out that convolutional networks and recurrent networks share a certain similarity, but also have an important difference. The similarity comes from shared weight matrices. In the case of convolutional networks, the same convolutional filter is applied to many locations of an image; for recurrent networks, we have the same "filter" being applied to every time step. The difference is that in the case of the convolutional network, errors propagate across layers, but not across images. On the other hand, with recurrent networks, errors propagate across layers *and* across series (a time series is the analog of an image when we talk about recurrent networks). Image applying the convolutional filter to each location in the image sequentially, but that each application depends on *all* the previous applications: the output of the filter in the bottom right corner of the image depends on the output from the upper left. The practical result of this difference is that the basic RNN cannot generally capture dependencies over relative long periods of time (LSTM and GRU networks can overcome this short-coming, but see this interesting [2019 ICML]() paper for an interesting take on long-term dependencies captured by LSTM).

Here is the common graphical depiction of a basic RNN cell:

![RNN cell](/assets/rnn_cell.jpg)

Now here is the "unrolled" version of the RNN network, where we draw a "copy" of the cell for each time step instead of the self-connection:

![RNN unrolled](/assets/rnn_unrolled.jpg)

It looks just like the graph model above! So a good way to think about the basic RNN model is as a sort of generalized state-space model in which the transition can and output nodes can depend on the state in a highly nonlinear fashion.


# Tensorflow Implementation
Tensorflow has methods to implement a variety of recurrent networks, but I'm going to write these up from scratch to help demonstrate what's going on (see this [gist]() for the complete code). I'm going to create a simple class to represent an RNN cell. I'm going to ignore the output ($$y$$) for now because that isn't really essentially to the cell, and you'll see that this makes life a bit easier/more flexible later on. The class basically just keeps track of the weights required to update the hidden state.

```python
class RNN(object):
    """Basic RNN cell."""

    def __init__(self, input_dim, hidden_dim, use_dropout=True):

        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wxh = tf.Variable(
            tf.truncated_normal([input_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_input'
        )
        self.Whh = tf.Variable(
            tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_hidden'
        )
        self.bh = tf.Variable(
            tf.constant(0.0, shape=[hidden_dim]),
            name='bias_hidden'
        )

        self.use_dropout = use_dropout
```

The core method for this class is a method that will construct the part of the graph representing a forward pass through the cell, taking data ($$x_t$$) and hidden state ($$s_{t - 1}$$) as input, and spitting out the new hidden state. But because we're going to train the network using sequences of characters, the `forward` method will actually return a whole sequence/list of hidden state tensors, as well as the last hidden state tensor (for convenience--it's the same as the last element of the list of course). There are really two strategies we could use. Here I'm going to add a bunch of tensors to the graph, one for each step of the input sequence (represented by `hidden` variable below). Alternatively, we could make `hidden` a `tf.Variable` (whose value can be modified--a tensor's cannot), but we have to be careful to specify that the variable is not trainable.

```python
def forward(self, inputs, init_hidden, keep_prob):
    """Construct graph for training RNN.

        # Arguments
        - `inputs`: list of [None, input_dim] placeholders
        - `hidden`: [None, hidden_dim] placeholder representing initial hidden state
        - `keep_prob (tf.placeholder)`: placeholder representing the dropout keep probability

        # Returns
        - `outputs`: list of [None, hidden_dim] tensors
        - `hidden`: [None, hidden_dim] tensor representing final hidden state
    """
    hidden = init_hidden
    outputs = []
    for input in inputs:
        hidden = tf.tanh(
            tf.add(
                tf.add(
                    tf.matmul(input, self.Wxh),
                    tf.matmul(hidden , self.Whh)
                ),
                self.bh
            )
        )
        if self.use_dropout:
            outputs += [tf.nn.dropout(hidden, keep_prob)]
        else:
            outputs += [hidden]
    return outputs, hidden
```

When it comes time to evaluate our network, we can of course calculate the loss on some hold-out data set, but ultimately we want to use this network to generate plausible sample text. When we generate text, we only provide a single initial character, let the model generate an output character, feed the output back into the network and repeat until we reach a given sample length. There are a few issues here. First, we need a new graph. The graph we made for training the network assumes a sequence of inputs and outputs, but here we just have a single input.

```python
def predict(self, input, hidden):
    """Construct graph for predicting next output from an input.

        # Arguments
        - `input`: [None, input_dim] placeholder
        - `hidden`: [None, hidden_dim] placeholder representing initial hidden state

        # Returns
        - `next_hidden`: [None, hidden_dim] tensor representing next hidden state
    """
    next_hidden = tf.nn.tanh(
        tf.add(
            tf.add(
                tf.matmul(input, self.Wxh),
                tf.matmul(hidden, self.Whh)
            ),
            self.bh
        )
    )
    return next_hidden
```

This looks exactly the same as one iteration of the loop in the `forward` method. The difference is that we're going to feed in distinct placeholders for `input` and `hidden`. The second issue is that we can't just generate logits anymore--we need to make an actual prediction to feed back into the network. That's not a big deal: we'll just sample a multinomial distribution based on logits. But instead of using the normal softmax function, we'll use a slight variation on it:

```python
def multinomial(input, weights, bias, temperature=1.0):
    """
        `temperature` -> 0 => argmax(logits)
    """
    logits = tf.matmul(input, weights) + bias
    return tf.multinomial(logits / temperature, 1)
```

Here we divide the logits by `temperature`, which causes the multinomial distribution to converge to either a uniform distribution when `temperature` goes to infinity, or else collapse onto the outcome with the maximum logit. We use this modified softmax to adjust the entropy (randomness) of the prediction distribution.

Finally, we want to be able to stack RNN cells together to form the equivalent of several convolutional layers. I'll make another simple class to help out. The class just keeps track of the individual cells--in order--as a list.

```python
class Chain(object):

    def __init__(self, cells):

        super(Chain, self).__init__()

        self.cells = cells
```

Now we can make `forward` and `predict` methods for `Chain` using the corresponding methods for the individual cells. The one "trick" is that the stacked cells now receive a list of hidden states--one for each cell--and return a list as well.

```python
def forward(self, inputs, init_hiddens, keep_prob):
    """
        # Arguments
        - `inputs (list<tf.placeholder>)`: `backprop_length` list of `[batch_size, input_dim]` placeholders
        - `init_hidden (list<tf.placeholder>)`: `len(cells)` list of `[batch_size, hidden_dim]` placeholders

        # Returns
        - `outputs (list<tf.tensor>)`: `backprop_length` list of `[batch_size, input_dim]` tensors
        - `hidden (list<tf.tensor>)`: `len(cells)` list of `[batch_size, hidden_dim]` tensors
    """

    hidden_list = []
    for (cell, init_hidden) in zip(self.cells, init_hiddens):
        outputs, hidden = cell.forward(inputs, init_hidden, keep_prob)
        inputs = outputs
        hidden_list += [hidden]
    return outputs, hidden_list

def predict(self, input, hiddens):
    hidden_list = []
    for (cell, hidden) in zip(self.cells, hiddens):
        hidden = cell.predict(input, hidden)
        input = output = hidden
        hidden_list += [hidden]
    return output, hidden_list
```

Basically the rest of the graph will just specify the loss and add a training operation to update the network weights.

# Example: HomeRNN
Here's an example where we train the character-level model using Homer's Iliad and Odyssey as the corpus. We'll compare a basic RNN cell with an LSTM cell. In each case, I'll layer up two cells, each having a hidden state of size 512. Here's are the training details:

```python
tf.app.flags.DEFINE_integer('batch_size', 64, """Sequences per batch.""")
tf.app.flags.DEFINE_integer('backprop_length', 64, """Training sequence length.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Learning rate.""")
tf.app.flags.DEFINE_integer('input_dim', 67, """Size of input.""")
tf.app.flags.DEFINE_integer('output_dim', 67, """Size of output.""")
tf.app.flags.DEFINE_integer('hidden_dim', 512, """Size of hidden state.""")
tf.app.flags.DEFINE_integer('sample_len', 512, """Size of text sample(s).""")
```

## Data
It's worth taking a minute to be clear on how we actually train these things.

I have The Iliad and The Odyssey stored as text files. Processing them for training is a matter of removing some unwanted additional text, splitting the text into individual characters, and converting the characters to unique integers (while keeping a record of how to convert integers back into characters for later use).

Next, we need to package the long list of integers into batches of training data, and here a picture is worth, well, about a paragraph I'd say in this case:

![Training data diagram](/assets/img/rnn_training_data.png)

In short, we split the data twice. First, we split it into `batch_size` *long* sequences, then we split each of these sequences up into `backprop_length` length subsequences (and we toss out any data that doesn't fit into our splits evenly). In this way our training batches end up as `batch_size x backprop_length x input_dim` tensors. For the character-level model, the `input_dim` is the number of unique characters in the corpus because we will represent the characters as one-hot vectors. This is reasonable for a character-level model because there are only 67 unique tokens; word-level models require us to be a bit more thoughtful.

## RNN
The image below shows the training curve for the RNN and LSTM models. The RNN model flattens out with a cross-entropy loss of 1.41 at around 10000 steps. We can get a sense of what's going on by looking text samples generated during the training loop.

#### step = 0, xentropy =  4.20
> (NMnYH'-dLBFzW]PtZkgEyF'vOuCvH?jx!OfcH(iI"bzYLKTokax:REGxO)aRCkr[&V)-[DTDOmlQ-lhhUUI!gE :PaakiNNjPHPeHKaN)p[GUt&mFW!ppwSbQglsF"[fjAic f&iiK fhLGF-WWFA eM:OyEF)QCSooAIgnRri&B:xhPcyO[YE?Z?oAnx[yazW&ec!asRpD'e"hytZyHDSE!AaUabtW"Ohb)[jUvc ,(tP[fuzcqw[p[ntX;--)WKJsHV-[j;ZzGNj[QaUPPfRclQ(xhrRSE?kI[&-!(xqfk!hh)DvFJFz GwZNkNHkBtf;.Z--]bm?vOVPnmHY]MV?xcGdewvOxcSWlRlC"]SA't'C]wVliiX"XC.pprP.fDXHJQuoM]KG]oy;ILUFS-,uoIskCn?Zb&sdc'Y"eX[?ZCPkmMEpfmTMV]EOnB(XmhR?y?LStyMo)f&f&WwI'YnPIIqoUPW[e[rgQ!?DFbP!g:,&'exC:vlRhNPE"wRz

Initially, the output is (obviously) random junk.

#### step = 200, xentropy =  2.44
> (ke cod the he the pos wod and he sis ho ho mat un the the the hi he the we tu  he the the se mir wo to the The ha wor wir wad hi the so the le we the we the wave be the he th the mn on the he won tha got s an th  ue af and tod no the the han ho yor wh the tone fo d she he the bos an the he bhe ff and ho tor the he hed f th and and ha the woud Ioc and soe the the ond ta gid the ghe bod on ho the hiu hot the and he s and Af of whe and fe the Iin he son bud aod an sa the the soos and the wire wo the se the bi

Not that long into training, the model is beginning to show signs of learning from the text sample alone. This is no longer a random sequence of characters. The model is correctly spelling shorter words like "he", "the", and "on", and it seems to have learned that all words are short--or it just "gives up" after a few characters and spits out a space.

#### step = 11180, xentropy =  1.41
> he deep that we take the ship and reary them. As all about the fleet son of Atreus, who was the far the son of Telamon bear the son of Tydeus the bow come to her and seat all. I will not kill my chariot and sheep and sent me on the sea who was the first to fight and said, "When we say which the son of Atreus and Achilles were distant with the banks of the Achaeans was the bow and with a stand from the ships to the gods was finsten to his sons of the Achaeans will and brought a councely and come to the son o

At the end of training, the model has learned to spell some fairly long words, and it's nailing proper nouns that show up constant in Homer ("Achaeans", "Atreus", and "Achilles") as well as some less common names ("Telamon" and "Tydeus"). It's learned how to start a quote ("said, "When...""), but it's still making some spelling mistakes ("reary", "finsten", and "councely"), and overall the text doesn't make any sense.

## LSTM
The LSTM model reaches a substantially lower training cross-entropy loss (under 1.00) after about an hour of training. Let's take a look at some sample text based on the final LSTM model.

#### step = 34890, xentrop = 0.87
> wally on your wife to shelt stand by me at the world love to Patroclus son of King Plespus, while he is in an irmortals, that we may be truly meanchist, and making me as saud as prayed my time, while I sirely trying and spare me not to make come to you my comrades at once. We will win bring up to me, for the band of Jove may come back with you. There is a man dogs be offering from your convoice; you have done so those that gave me twelve minds which I will make it master of what I may turn here to be told

The LSTM eventually does a substantially better job predicting the next character, but the sample text doesn't seem much difference. In these two small (~500 characters) samples, there are actually slightly more spelling errors in the LSTM sample than the RNN sample. I trained these using a GTX 1080 GPU using batches of 64 sequences, with 64 characters per sequence. The average time to process a batch was around 0.04 seconds for the RNN, and about 0.13 seconds for the LSTM model.

![RNN vs LSTM](/assets/img/rnn_vs_lstm.png)

# Parting Thoughts
I would be remiss if I didn't mention in closing that this isn't really a good language model. Consider for a moment that this model is not even possible in Chinese (no alphabet), or sign language. Words are the fundamental unit of language; letters are tools for the written externalization of language. On the other hand, a computer that didn't have a character-level model might not be able to really master the English language in the sense that it wouldn't be able to create new words. In other words, there are really two parts to what we colloquially think of language: thought and externalization of thought. This is perhaps a good model of the latter, but not the prior. For that, we need to consider *word-level* models, which I'll look at next time.
