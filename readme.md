# tiny-llm

A tiny GPT-style chatbot written from scratch in TypeScript, with no ML libraries. It learns a small set of conversations one character at a time, and every step of training and generation is plain loops you can read.

## Quickstart

```sh
npm install
npm run train      # trains on data.txt and writes model-weights.json (takes a few minutes)
npm run chat       # chat with the bot
npm run generate   # lets the model write a whole conversation on its own
npm run typecheck  # type-checks all files
```

## Files

| File                 | What it does                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `data.txt`           | The training conversations. Edit this to change what the bot says.                                         |
| `model.ts`           | The shared model: math helpers, forward pass, sampling, generation and loading weights.                    |
| `1-train.ts`         | Reads `data.txt`, initialises the weights, runs the training loop (backpropagation) and saves the weights. |
| `2-use-llm.ts`       | Starts a conversation with `user: hi` and lets the model write the rest, both sides.                       |
| `3-chat.ts`          | A terminal chat. Type a message, or `exit` to quit.                                                        |
| `model-weights.json` | The trained model, written by `npm run train`.                                                             |

## The chat format

Every exchange in `data.txt` looks like this:

```text
user: what is your name?
bot: i am tiny gpt.
```

The model never "knows" it's in a chat. It only learns which character tends to come next. Because every `bot:` line in the data follows a `user:` line, it learns that `bot:` is followed by an answer to the line before it.

When you type a message, `3-chat.ts`:

1. Lowercases it and removes any characters that never appear in `data.txt`.
2. Adds it to a running transcript of the conversation, in the same format: `user: <your message>\nbot: `.
3. Lets the model continue from there and stops at the end of the line, so the model doesn't go on to write your next message too.
4. Adds the reply to the transcript.

The transcript starts with a hidden opening exchange (`user: hi` / `bot: hello, how are you?`). In training, almost every question had the end of the previous answer in front of it, filling the 32-character window. Keeping a transcript, with that opening exchange, means questions in the chat look the same, so every character sits in the position the model learned it in. Without it, short windows only ever meant the very start of `data.txt`, and the bot tried to answer everything with "hello".

The bot can only see the last 32 characters. Keep questions short, or the start of the question falls out of view before the answer begins. The transcript doesn't give the bot a real memory: it only ever sees the last 32 characters, which is your question plus a little of its previous answer.

## How the Mini-GPT works

The model reads the last 32 characters (the **context window**) and predicts the next one. One prediction looks like this:

```text
 "...what is your name?\nbot: "     last 32 characters
     │
     ▼
 character IDs         w, h, a, t, ...
     │
     ▼
 token embedding       tokenEmbed[id]  +  posEmbed[position]
     │
     ├──────────────────────────┐
     ▼                          │
 layer norm                     │
     │                          │
     ▼                          │
 self-attention                 │
   Q, K, V = N · Wq, Wk, Wv     │
   last Q compared with all K   │
   → attention weights          │
   → weighted sum of V          │
     │                          │
     ▼                          │
 residual add  ◄────────────────┘  (the last character's own embedding)
     │
     ├──────────────────────────┐
     ▼                          │
 layer norm                     │
     │                          │
     ▼                          │
 MLP                            │
   expand with W1 (32 → 128)    │
   ReLU (negatives become 0)    │
   shrink with W2 (128 → 32)    │
     │                          │
     ▼                          │
 residual add  ◄────────────────┘
     │
     ▼
 layer norm
     │
     ▼
 Wout → one score per character
     │
     ▼
 softmax → probabilities → weighted random pick → next character
```

1. **Tokenizer.** Neural networks only do math on numbers, so every unique character in the training text gets an ID. Real LLMs use tokens (chunks like "ing" or "the"); this model keeps it simple with single characters.
2. **Embeddings.** Each character ID becomes a 32-number vector (`tokenEmbed`). A second vector for its position in the window (`posEmbed`) is added, so the model knows the order of the characters.
3. **Layer normalisation.** Each vector is rescaled to a standard size (average 0, spread 1) before attention, before the MLP and before the output layer. Without it, the vectors keep growing during training until it becomes unstable: the loss starts rising or turns into `NaN`. Real GPTs normalise in the same places.
4. **Self-attention.** Each position produces a Query, Key and Value (`Wq`, `Wk`, `Wv`). The last character's Query is compared with every Key to decide how much attention to pay to each earlier character, and the Values are blended using those attention weights. This is how the model uses context: when writing the answer after `bot: `, it can look back at the words in the question.
5. **Residual connection.** The last character's own embedding is added to the attention result, so the model always sees the current character directly as well as its context.
6. **MLP (feed-forward layer).** The vector is widened to 128 numbers (`W1`), every negative number is set to 0 (ReLU), and it's shrunk back to 32 (`W2`). The result is added on top, with a second residual connection. Attention gathers information from the context; the MLP is where most of what the model has memorised is stored, such as which answer goes with which question. Together, attention + MLP make one **transformer block**, the building block of every GPT.
7. **Output & softmax.** `Wout` turns the normalised result into one raw score per character. Softmax turns those scores into probabilities that add up to 100%.
8. **Sampling.** The next character is picked by rolling a weighted die over the probabilities, rather than always taking the most likely one. This stops the model from getting stuck in loops. **Temperature** controls how adventurous it is: below 1.0 favours the most likely characters, and 1.0 samples the learned probabilities as they are.

### Training

`1-train.ts` goes through every position in `data.txt`, in a new random order each epoch. At each position it gives the model the previous characters (up to 32) and asks it to predict the next one.

- **Loss.** How wrong the guess was is measured with cross-entropy: `-log(probability given to the correct character)`. It starts around 3.0–3.5, which is random guessing over all the characters, and falls as the model learns.
- **Backpropagation.** The backward pass works out how every weight contributed to the error. It goes through the output layer, the MLP (including the ReLU), all three layer norms, the attention softmax, the Query/Key/Value matrices, the residual path and back into both embedding tables. All gradients are computed from the current weights before any of them are updated.
- **Gradient descent.** Every weight is then nudged a little in the direction that reduces the error (`weight -= learningRate * gradient`). Each gradient is first clipped to ±1 (`gradClip`), so one unusually large gradient can't knock a weight far off. The learning rate starts at 0.01 and is lowered steadily to 0.001 by the last epoch: big steps learn fast early on, and small steps late in training avoid sudden spikes that can undo what was learned.
- **Best weights.** Whenever an epoch's loss is the lowest so far, a copy of the weights is kept, and that copy is what gets saved, not the last epoch's. A late spike in the loss can't ruin the saved model. Training prints which epoch the saved weights came from.
- **Safety net.** If the loss ever becomes `NaN`, training stops without saving, so a failed run never overwrites your weights. Repeating this 1000 times over the text is how the model learns. It's the same idea used to train every neural network, including real LLMs.

### Settings

| Setting                         | Value          | Where          |
| ------------------------------- | -------------- | -------------- |
| Training data                   | `data.txt`     | `1-train.ts`   |
| Context window (`blockSize`)    | 32 characters  | `1-train.ts`   |
| Embedding size (`embedDim`)     | 32             | `1-train.ts`   |
| MLP size (`mlpDim`)             | 128            | `1-train.ts`   |
| Learning rate                   | 0.01 → 0.001   | `1-train.ts`   |
| Gradient clipping (`gradClip`)  | ±1             | `1-train.ts`   |
| Epochs                          | 1000           | `1-train.ts`   |
| Loss printed every (`logEvery`) | 50 epochs      | `1-train.ts`   |
| Generate temperature            | 1.0            | `2-use-llm.ts` |
| Chat temperature                | 0.25           | `3-chat.ts`    |
| Longest chat reply              | 100 characters | `3-chat.ts`    |

`blockSize` and `embedDim` are saved with the weights, so `2-use-llm.ts` and `3-chat.ts` pick them up automatically.

## Saved weights

`npm run train` writes `model-weights.json`:

```jsonc
{
  "config": { "version": 4, "blockSize": 32, "embedDim": 32, "vocabSize": 31 },
  "vocab": ["\n", " ", ",", ".", ":", "?", "a", ...], // character for each ID
  "trainingText": "user: hi\nbot: hello, ...",       // the data it was trained on
  "weights": {
    "tokenEmbed": [...], // [vocabSize][embedDim]
    "posEmbed":   [...], // [blockSize][embedDim]
    "Wq": [...], "Wk": [...], "Wv": [...], // [embedDim][embedDim]
    "W1": [...],         // [embedDim][mlpDim]
    "W2": [...],         // [mlpDim][embedDim]
    "Wout": [...]        // [embedDim][vocabSize]
  }
}
```

`version` is checked when the weights are loaded. If the model architecture in `model.ts` changes, bump `MODEL_VERSION` so old weight files are rejected with a "retrain" error instead of silently producing gibberish.

## Experimenting

Change something, then run `npm run train` again:

- **More conversations.** Add `user:` / `bot:` pairs to `data.txt`. Everything is lowercased when it's loaded. Every new character you use (digits, `!`, `'`) becomes part of the vocabulary. More data means better variety but slower training.
- **Learning rate.** `startLearningRate` and `endLearningRate`. Higher learns faster but can overshoot. If the loss climbs or turns into `NaN`, lower them.
- **Epochs.** More passes lower the loss further. On this small dataset the model eventually memorises the answers word for word.
- **Context window and embedding size.** A larger `blockSize` lets the model see longer questions. A larger `embedDim` gives it more capacity. Both make training slower.

Expect a parrot, not a thinker: the bot answers questions it was trained on (or close to them) well, and mixes bits of answers together for anything else.

Epoch
An epoch is one full pass through all of data.txt.

Within one epoch, the model goes through the text character by character. At each position it sees the previous characters (up to 32) and guesses the next one. Then it checks the right answer and adjusts its weights a little. Your data is about 2,500 characters, so one epoch is about 2,500 guesses and 2,500 small adjustments.

One pass isn't enough to learn anything, so training repeats this epochs = 1000 times (1-train.ts). Each pass, the guesses get a bit better. The number printed is which pass it has reached, counting from 0, so Epoch 999 is the last one.

Error loss
The loss measures how wrong the model's guesses were during that epoch. Lower is better, and 0 would mean perfect.

For each guess, the model assigns a probability to every possible next character. The loss looks only at the probability it gave to the correct character:

loss = -log(probability given to the right answer)
Probability given to the right character Loss
100% (certain and correct) 0.00
90% 0.11
50% 0.69
10% 2.30
3% (about random with 31 characters) 3.43
The printed number is the average over every guess in that epoch.

What the numbers mean for your training
About 3.4 at epoch 0: pure guessing. With 31 possible characters, a random guess gets the right one about 1 time in 31.
About 2.0–2.5: it has learned simple patterns: which letters are common, that spaces come between words, and that bot: comes after user: lines.
About 1.0: it's getting most of each word right.
Below about 0.3: it has memorised most answers and is usually confident and correct.
If the loss stops falling, the model has learned as much as it can at this size. If it goes up or shows NaN, the weight adjustments are too large, and lowering learningRate fixes that.

Here are the specs of your model as it is now.

| Detail                       | Your model                                         |
| ---------------------------- | -------------------------------------------------- |
| **Parameters**               | **14,272** (about 14 thousand, or 0.000014B)       |
| Layers (transformer blocks)  | 1                                                  |
| Attention heads              | 1                                                  |
| Embedding size (`embedDim`)  | 32                                                 |
| MLP size (`mlpDim`)          | 128                                                |
| Context window (`blockSize`) | 32 characters                                      |
| Vocabulary                   | 31 characters (letters, space, newline, `, . : ?`) |
| Tokenizer                    | One character per token                            |
| Training data                | `data.txt`, about 2,500 characters (43 exchanges)  |
| Training                     | 1000 epochs, about 2.5 million weight updates      |

**Where the parameters are**

| Weights          | Shape         | Count      |
| ---------------- | ------------- | ---------- |
| `tokenEmbed`     | 31 × 32       | 992        |
| `posEmbed`       | 32 × 32       | 1,024      |
| `Wq`, `Wk`, `Wv` | 3 × (32 × 32) | 3,072      |
| `W1` (MLP)       | 32 × 128      | 4,096      |
| `W2` (MLP)       | 128 × 32      | 4,096      |
| `Wout`           | 32 × 31       | 992        |
| **Total**        |               | **14,272** |

The layer norms and the rest of the model have no learnable parameters (no biases either).

**For comparison**

| Model              | Parameters  | Times bigger than yours |
| ------------------ | ----------- | ----------------------- |
| Your model         | 14 thousand | 1×                      |
| GPT-2 small (2019) | 124 million | about 8,700×            |
| GPT-3 (2020)       | 175 billion | about 12 million×       |

Modern chatbots are also trained on trillions of tokens of text, compared with your 2,500 characters. That's why yours can only repeat what's in `data.txt`.
