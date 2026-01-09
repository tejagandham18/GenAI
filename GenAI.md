# 🧠 Artificial Intelligence Notes — Day 1 & Day 2

---

## 📌 **What is Human Intelligence?**

Human intelligence is the natural ability of humans to think, understand, learn, reason, and solve problems.  
It involves creativity, decision-making, emotional understanding, and the capacity to adapt to new situations.  

It is influenced by:
- biological brain function  
- experiences  
- environment  
- culture  

Human intelligence is **flexible, conscious, and multi-dimensional**, enabling communication, social interaction, and innovation in the real world.

---

## 🤖 **What is a Neural Network?**

A neural network is a computational model inspired by the human brain.

At the beginning, the neural network knows nothing.

When we give it data, the first thing it does is **accept inputs** — these inputs can be numbers like age, salary, study hours, or even pixels of an image.

Each input is not treated equally.  
The network assigns an importance value, called a **weight**, to every input.  
Important inputs get higher weights, less important ones get lower weights.

Inside the network, these inputs and weights are **combined and processed**.

This processing happens in the **hidden layers**, where the network tries to understand patterns in the data.

After processing, the network reaches a decision point using an **activation function**, which decides whether the signal should be strong or weak.

Finally, the network produces an **output**, which can be a prediction, a classification, or a probability.

If the output is wrong, the network **calculates error**, compares with the correct answer, and sends the error backward through the network.  
This backward step is used to **adjust weights** so performance improves next time.

This cycle repeats with more data.  
Over time, the neural network **learns from mistakes** and becomes more accurate.

In short:

> A neural network learns patterns through repeated experience and correction.

---

## 🧱 **What is Deep Learning?**

Deep learning builds on the same foundation as a neural network, but instead of having one or two hidden layers, deep learning uses **dozens or even hundreds of hidden layers**.

This depth allows the model to learn **multiple levels of abstraction**.

In traditional machine learning, humans needed to **manually engineer features**.  
For example, for images we had to extract:
- edges
- textures
- shapes

Deep learning removes that manual step and learns these features **automatically** during training.

### 🔁 **Hierarchy Example (Image Processing)**

As data passes deeper:

- Early layers learn edges + lines  
- Middle layers learn ears + eyes  
- Deeper layers learn full objects like **“cat”**

After producing output, the model computes an error and applies **backpropagation** to adjust weights and improve accuracy.

This process:
> prediction → error → weight adjustment

is repeated across many examples.

Deep learning can learn directly from **raw, unstructured data** such as:
- images
- audio
- video
- text

In simple terms:

> Deep learning is an extension of neural networks to many layers, enabling automatic learning of complex patterns from data.

---

# 📅 **DAY 2**

---

## 🔤 **What is a Token?**

In language models, a **token** is the smallest unit of text that the model can read, understand, and process.

Tokens may represent:
- a whole word
- part of a word
- punctuation
- whitespace

Example:

Sentence:  
> “Apple is green.”

Tokenized as:  
`Apple`, `is`, `green`, `.` → **4 tokens**

Each token is converted into a **numerical ID**:  
`[5271, 181, 4490, 13]`

LLMs work internally on **numbers**, not text.

During generation, the model predicts **one token at a time** based on previous tokens.

Tokenization affects:
- cost
- speed
- context memory
- language flexibility

Modern LLMs have token context windows (e.g., 128K tokens), which determine how much text they can remember at once.

---

## 🧠 **What is a Transformer?**

A **Transformer** is a deep learning architecture used to understand and generate language.

Transformers look at **all words at the same time**, unlike older sequential models.

The core idea is **self-attention**, which helps identify important relationships between words.

Example:  
Sentence:  
> “The dog chased the cat because it was scared.”

Transformer can infer that **“it”** refers to **the cat**.

Transformers convert sentences into tokens and then into numbers before processing.

Example task:  
Input: `"Good morning"` (English)  
Output: `"Bonjour"` (French)

Transformers are used in:
- ChatGPT
- BERT
- GPT-4
- Gemini
- Claude

They are efficient at handling text, Q&A, reasoning, and even images/audio.

---

## 🧩 **How Different LLMs and Tokenization Work**

LLMs do not process full sentences like humans.  
They break text into **tokens** first.

Example:  
> “Artificial intelligence is powerful.”

Tokenized as:  
`Artificial`, `intelligence`, `is`, `power`, `ful`, `.`

---

### **Different LLMs Use Different Tokenization Methods**

**A. GPT Models (OpenAI) → BPE (Byte Pair Encoding)**  
Example:  
`Internationalization` → `Intern`, `ation`, `al`, `ization`

**B. Google Gemini / T5 → SentencePiece**  
Example:  
`Artificial Intelligence` → `▁Artificial`, `▁Intel`, `ligence`

**C. Meta LLaMA → BPE Multilingual Variant**  
Example:  
`Computational` → `Compute`, `ation`, `al`

**D. Anthropic Claude → BPE Variant**  
Optimized for long documents.

**E. Chinese/Japanese/Korean Models → Character Tokenization**  
Example (Chinese):  
`我爱你` → `我`, `爱`, `你`

---



