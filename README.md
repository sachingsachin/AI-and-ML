# AI-and-ML

## Some common techniques for Gen-AI

A. Generative Adversarial Networks (GANs)
B. Variational Autoencoders (VAEs)
C. Transformer Models

---

### A. Generative Adversarial Networks (GANs):
GANs consist of two neural networks, a generator, and a discriminator, that compete against each other. The generator creates new data samples, and the discriminator evaluates them against real data, improving the generator's performance over time.

---

### B. Variational Autoencoders (VAEs):
VAEs are used for generating new data points by encoding data into a latent space and then decoding it back into the original data space.

**1. Encoder**:  
The VAE consists of an encoder network that compresses input images into a smaller, latent space representation. This latent space is typically a distribution (e.g., a Gaussian distribution) rather than a fixed point. For example, an input image of a digit "5" is encoded into a **mean vector** and a **variance vector** representing the latent distribution.

**2. Latent Space Sampling**:  
From the latent space, we sample a point (latent vector) from the learned distribution. This sampled point represents the compressed version of the input image.

**3. Decoder**:  
The decoder network takes the sampled latent vector and reconstructs the image from it. The goal is to generate an image that closely resembles the original input image. This is often done through a series of upsampling and convolutional layers, especially in the case of image data.

**4. Evaluation**:  
The generated image is compared to the original input image using a loss function. Typically, the reconstruction loss (such as mean squared error or binary cross-entropy) measures how well the generated image matches the original one.

**5. Backpropagation and Training**: The total loss, which includes both the reconstruction loss and the KL divergence loss, is used to update the weights of the encoder and decoder networks through backpropagation. This iterative process continues during training to improve the accuracy and quality of the generated data.

**6. Generation of New Data**: Once the VAE is trained, it can be used to generate new data samples by sampling latent vectors from the learned distribution and decoding them into the data space. This allows for the creation of novel images that resemble the training data but are unique.

Applications: The generated data can be used in various applications, such as data augmentation, creative content generation, anomaly detection, and more. By effectively encoding, sampling, and decoding, VAEs enable the generation of new, realistic data points that can be used in diverse applications.

**Key Components of VAEs**  
1. Reconstruction Loss: Measures how well the generated image matches the original input image. This loss encourages the decoder to produce accurate reconstructions.

2. KL Divergence Loss: Measures the difference between the learned latent space distribution and a prior distribution (e.g., a standard Gaussian distribution). This loss encourages the latent space to be smooth and continuous.

---

### C. Transformer Models:
Models like GPT-3 (Generative Pre-trained Transformer 3) and BERT (Bidirectional Encoder Representations from Transformers) are widely used for generating text and understanding natural language. GPT-2, the predecessor of GPT-3, has 1.5 billion parameters. The leap to 175 billion parameters in GPT-3 represents a significant increase in model complexity and capability. GPT-3 is based on neural networks, specifically a type known as transformer networks. GPT-4 onwards, the number of parameters is not publicly disclosed but its estimated that it has crossed the 1 trillion parameter mark.

**Nodes (or Neurons) in Neural Networks**  
1. Nodes: In neural networks, nodes (or neurons) are the basic units of computation. Each node processes inputs, applies a function, and passes the output to the next layer of nodes.
2. Layers: Neural networks are composed of layers of nodes. There are input layers, hidden layers, and output layers. In transformer models like GPT-3, these layers include `self-attention` and `feed-forward` layers.

**Parameters in Neural Networks**
1. Parameters: In neural networks, parameters refer to the weights and biases that determine the strength and nature of the connections `between the nodes`. These parameters are adjusted during training to minimize the error in the model's predictions.
2. Representation: Each parameter is not directly represented by a single node. Instead, the parameters are associated with the connections between nodes and the nodes' computations. In other words, **a node may have multiple incoming and outgoing connections, each with its own weight parameter**.

GPT-3 uses the transformer architecture, which includes multiple layers of `self-attention mechanisms` and `feed-forward` neural networks. The self-attention mechanism allows the model to weigh the importance of different words in a sequence and capture long-range dependencies.

---

## MoE - Mixture of Experts - How DeepSeek achieves drasticallty lower cost

MoE model does not have a single uber-expert-of-everything. Rather it has several expert models and it invokes the right one by understading the query. Due to this, a single model need not be trained on everything and the parameter space handled by each model is reduced drastically. Example: there could be lawyer model and a doctor model and queries are sent to either one of those based on type of query.


## Transformer Model

The name Transformer comes from the model’s ability to transform an input sequence into an output sequence using only attention mechanisms — without recurrence (RNNs) or convolutions (CNNs).

In the original paper (“Attention Is All You Need”, 2017), the task was sequence-to-sequence translation.

Example: input = English sentence, output = French sentence.

The encoder transforms the input sequence into a **contextual representation**.

The decoder transforms that into an output sequence in the target language.

So the “transformation” is literally:  
Input tokens → contextual embeddings → output tokens

**What is being transformed?**

Embeddings: Raw word/token embeddings are transformed layer by layer into richer contextual representations.

Relationships: Self-attention transforms the representation of a token by mixing in weighted information from all other tokens.

Example: the representation of “sat” is transformed to include context from “cat” (who sat) and “mat” (where it sat).

Sequences: An entire input sequence is transformed into an output sequence (translation, summarization, next-word prediction, etc.)


**All the steps of the transformer model**

1. Tokenize a sentence.
2. Capture positional information for each token.
3. Activate self-attention: This process uses the position information to draw relationships between tokens. Like verb, article, preposition, subject, object etc. Based on this self-attention, it assigns weights to relationships between the tokens. 
4. Activate multi-head attention: Runs several self-attention mechanisms in parallel. A single self-attention head may only capture one type of relationship. Multiple heads let the model learn different relationships simultaneously. Say you have 8 heads. Each head computes its own attention output. These outputs are concatenated and linearly transformed into the final representation.

Example (“The cat sat on the mat”):

Head 1 might focus on subject–verb (cat → sat).

Head 2 might focus on preposition–object (on → mat).

Head 3 might focus on article–noun (the → cat).

Together, multi-head attention captures a richer picture than any single head.


