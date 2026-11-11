import * as fs from "fs";

// ==========================================
// Shared Mini-GPT model code (used by training, generation and chat)
// ==========================================

/**
 * A 2D grid of numbers, stored as an array of rows: `mat[row][col]`.
 * Every weight in the model is stored as a Matrix.
 */
export type Matrix = number[][];

/**
 * Everything the model learns during training: its "brain".
 *
 * These six matrices start as small random numbers, and training nudges
 * them until the model predicts the next character well.
 */
export interface Weights {
  /** One vector per character: what each character "means" to the model. [vocabSize][embedDim] */
  tokenEmbed: Matrix;
  /** One vector per position in the context window, so the model knows character order. [blockSize][embedDim] */
  posEmbed: Matrix;
  /** Query matrix: turns a character into the question "what am I looking for?". [embedDim][embedDim] */
  Wq: Matrix;
  /** Key matrix: turns a character into the label "this is what I contain". [embedDim][embedDim] */
  Wk: Matrix;
  /** Value matrix: turns a character into the information it passes on when attended to. [embedDim][embedDim] */
  Wv: Matrix;
  /** MLP expand matrix: widens the vector to 4× its size, where patterns are stored. [embedDim][4 × embedDim] */
  W1: Matrix;
  /** MLP shrink matrix: brings the widened vector back to embedDim. [4 × embedDim][embedDim] */
  W2: Matrix;
  /** Output matrix: turns the final vector into one score per character in the vocabulary. [embedDim][vocabSize] */
  Wout: Matrix;
}

/**
 * The size settings of a trained model. They're saved with the weights so the
 * generate and chat scripts use the same sizes the model was trained with.
 */
export interface ModelConfig {
  /** Architecture version the weights were trained with (see MODEL_VERSION). */
  version: number;
  /** Context window: how many previous characters the model can look at. */
  blockSize: number;
  /** Length of each character's vector (embedding). Bigger = more capacity, slower. */
  embedDim: number;
  /** Number of unique characters the model knows. */
  vocabSize: number;
}

/**
 * The exact shape of model-weights.json, as written by 1-train.ts.
 */
export interface SavedModel {
  /** Size settings the model was trained with. */
  config: ModelConfig;
  /** The vocabulary: vocab[id] is the character with that ID. */
  vocab: string[];
  /** The original training text, kept for reference. */
  trainingText: string;
  /** The learned weights. */
  weights: Weights;
}

/**
 * A loaded model, ready to use: the saved model plus lookup tables
 * for converting between characters and IDs.
 */
export interface Model extends SavedModel {
  /** Character → ID (the tokenizer), e.g. "a" → 2. */
  charToId: Map<string, number>;
  /** ID → character (turns predictions back into text), e.g. 2 → "a". */
  idToChar: Map<number, string>;
}

/**
 * Version number of the model architecture.
 *
 * Bump this whenever the forward pass changes in a way that makes old weights
 * meaningless. loadModel() then rejects old weight files with a clear
 * "retrain" error instead of silently producing gibberish.
 *
 * v2: residual connection + full embedding gradient.
 * v3: layer normalisation before attention and before the output layer.
 * v4: MLP (feed-forward) block after attention.
 */
export const MODEL_VERSION = 4;

/** File name the trained weights are saved to and loaded from. */
export const WEIGHTS_FILE = "model-weights.json";

// ==========================================
// Math helpers
// ==========================================

/**
 * Multiplies a vector by a matrix: `vec · mat`.
 *
 * This is how a layer transforms a vector: each output number is a weighted
 * sum of all the input numbers, and the matrix holds those weights.
 *
 * Example: a 16-number character vector · Wout (16 × 20) → 20 scores,
 * one per character.
 *
 * @param mat weights, shaped [inputSize][outputSize]
 * @param vec input vector, length inputSize
 * @returns output vector, length outputSize
 */
export function matVecMul(mat: Matrix, vec: number[]): number[] {
  const outDim = mat[0].length;
  const out = new Array(outDim).fill(0);
  for (let i = 0; i < vec.length; i++) {
    for (let j = 0; j < outDim; j++) {
      out[j] += vec[i] * mat[i][j];
    }
  }
  return out;
}

/**
 * Multiplies a matrix by a vector the other way round: `mat · vec`.
 *
 * Used during training (backpropagation). If a layer computed
 * `out = in · mat`, this takes the error on `out` and works out how much
 * of that error came from each number in `in`. It is matVecMul run backwards.
 *
 * @param mat weights, shaped [inputSize][outputSize]
 * @param vec gradient (error) on the output, length outputSize
 * @returns gradient on the input, length inputSize
 */
export function matVecMulT(mat: Matrix, vec: number[]): number[] {
  return mat.map((row) => row.reduce((acc, m, j) => acc + m * vec[j], 0));
}

/**
 * Turns raw scores into probabilities that add up to 1 (100%).
 *
 * Bigger scores get bigger probabilities. Each score is divided by
 * `temperature` first: below 1.0 makes the top choice even more likely
 * (more confident), above 1.0 flattens the odds (more random).
 *
 * The largest score is subtracted before Math.exp() so very large
 * scores can't overflow to Infinity. This doesn't change the result.
 *
 * Example: softmax([2, 1, 0]) → [0.67, 0.24, 0.09]
 *
 * @param logits      raw scores, one per option
 * @param temperature how sharp the probabilities are (default 1.0)
 * @returns probabilities in the same order as `logits`
 */
export function softmax(logits: number[], temperature = 1.0): number[] {
  const scaled = logits.map((x) => x / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Layer normalisation: rescales a vector so its numbers have an average of 0
 * and a spread (standard deviation) of 1.
 *
 * Without this, the vectors inside the model can keep growing during
 * training. Attention multiplies vectors together, so its scores grow even
 * faster, until training becomes unstable (the loss rises or turns into NaN).
 * Normalising keeps every vector the same size, whatever the weights do.
 * Real GPTs normalise in exactly these places.
 *
 * Example: layerNorm([1, 2, 3]).y → [-1.22, 0, 1.22]
 *
 * @param x the vector to normalise
 * @returns `y`, the normalised vector, and `invStd` (1 / standard deviation),
 *          which training needs to send gradients back through this step
 */
export function layerNorm(x: number[]): { y: number[]; invStd: number } {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const variance = x.reduce((acc, v) => acc + (v - mean) ** 2, 0) / x.length;
  const invStd = 1 / Math.sqrt(variance + 1e-5); // 1e-5 avoids dividing by zero
  return { y: x.map((v) => (v - mean) * invStd), invStd };
}

/**
 * Sends a gradient back through layerNorm (used in training).
 *
 * Given the gradient on layerNorm's output, returns the gradient on its
 * input. Because every output depends on the mean and spread of ALL the
 * inputs, each input's gradient depends on all the output gradients:
 *
 *   dx = invStd × (dy − mean(dy) − y × mean(dy × y))
 *
 * @param y      the output layerNorm produced
 * @param invStd the invStd layerNorm returned
 * @param dy     the gradient on the output
 * @returns the gradient on the input
 */
export function layerNormBackward(
  y: number[],
  invStd: number,
  dy: number[],
): number[] {
  const n = y.length;
  const meanDy = dy.reduce((a, b) => a + b, 0) / n;
  const meanDyY = dy.reduce((acc, g, i) => acc + g * y[i], 0) / n;
  return dy.map((g, i) => invStd * (g - meanDy - y[i] * meanDyY));
}

// ==========================================
// Forward pass: single-head causal self-attention for the last token
// ==========================================

/**
 * Runs the model once: given the characters in the context window,
 * returns the probability of every possible next character.
 *
 * Steps:
 *  1. Embed: each character's vector + its position's vector → X
 *  2. Normalise: layerNorm each position's vector → N
 *  3. Q, K, V: transform every normalised position with Wq, Wk, Wv
 *  4. Attention: compare the LAST character's query with every key,
 *     softmax the scores into attention weights, and blend the values
 *     with them → contextVec
 *  5. Residual: add the last character's own (un-normalised) vector
 *     back in → hidden
 *  6. MLP: layerNorm(hidden) → expand with W1 → ReLU (negatives become 0)
 *     → shrink with W2, then add back onto hidden (a second residual) → hidden2
 *  7. Normalise again: layerNorm(hidden2) → H
 *  8. Output: H · Wout → one score per character → softmax → probs
 *
 * All the in-between results are returned too, because training needs
 * them to work out the gradients (see 1-train.ts).
 *
 * @param w           the model weights
 * @param tokenIds    character IDs in the context window (1 to blockSize of them)
 * @param temperature applied to the final character probabilities only,
 *                    never to the attention (default 1.0)
 * @returns `probs` (next-character probabilities) plus the intermediate values
 */
export function forward(w: Weights, tokenIds: number[], temperature = 1.0) {
  const embedDim = w.Wq.length;

  // Token embedding + positional embedding
  const X: Matrix = tokenIds.map((id, pos) =>
    w.tokenEmbed[id].map((val, d) => val + w.posEmbed[pos][d]),
  );

  // Normalise each position before attention
  const normX = X.map((x) => layerNorm(x));
  const N: Matrix = normX.map((n) => n.y);
  const nInvStd = normX.map((n) => n.invStd);

  const Q = N.map((n) => matVecMul(w.Wq, n));
  const K = N.map((n) => matVecMul(w.Wk, n));
  const V = N.map((n) => matVecMul(w.Wv, n));

  const lastIdx = X.length - 1;
  const scale = Math.sqrt(embedDim);
  const rawScores = K.map(
    (k) => k.reduce((acc, kd, d) => acc + Q[lastIdx][d] * kd, 0) / scale,
  );
  const attnWeights = softmax(rawScores);

  // Weighted sum of values
  const contextVec = new Array(embedDim).fill(0);
  for (let t = 0; t <= lastIdx; t++) {
    for (let d = 0; d < embedDim; d++) {
      contextVec[d] += attnWeights[t] * V[t][d];
    }
  }

  // Residual connection: the last token's own embedding is added back in,
  // so the model always "sees" the current character directly.
  const hidden = X[lastIdx].map((x, d) => x + contextVec[d]);

  // MLP (feed-forward) block: normalise, expand with W1, ReLU, shrink with W2,
  // then add the result back onto hidden (its own residual connection).
  // This is where a transformer stores most of what it has memorised.
  const { y: M, invStd: mInvStd } = layerNorm(hidden);
  const mlpPre = matVecMul(w.W1, M);
  const mlpAct = mlpPre.map((v) => Math.max(0, v)); // ReLU
  const mlpOut = matVecMul(w.W2, mlpAct);
  const hidden2 = hidden.map((h, d) => h + mlpOut[d]);

  // Normalise again before the output layer
  const { y: H, invStd: hInvStd } = layerNorm(hidden2);

  const probs = softmax(matVecMul(w.Wout, H), temperature);
  return {
    X, N, nInvStd, Q, K, V, attnWeights, contextVec, hidden,
    M, mInvStd, mlpPre, mlpAct, hidden2, H, hInvStd, probs,
  };
}

// ==========================================
// Sampling & generation
// ==========================================

/**
 * Picks a random index, weighted by the probabilities: rolls a weighted die.
 *
 * A character with 70% probability is picked about 70% of the time.
 * Picking randomly instead of always taking the most likely character
 * stops the model from getting stuck repeating itself.
 *
 * @param probs probabilities that add up to 1
 * @returns the chosen index (a character ID)
 */
export function sample(probs: number[]): number {
  const rand = Math.random();
  let cumulative = 0;
  for (let j = 0; j < probs.length; j++) {
    cumulative += probs[j];
    if (rand <= cumulative) return j;
  }
  return 0;
}

/**
 * Writes new text, one character at a time.
 *
 * Each step takes the last blockSize characters, runs the model to get
 * next-character probabilities, picks one with sample(), appends it, and
 * repeats. Characters the model doesn't know are treated as ID 0.
 *
 * If `stop` is given, generation ends as soon as the new text ends with it,
 * e.g. stop = "\n" ends the reply at the end of the line.
 *
 * @param model       a loaded model (from loadModel)
 * @param context     the text to continue from
 * @param maxChars    the most new characters to write
 * @param temperature below 1.0 = safer and more repetitive,
 *                    1.0 = the learned probabilities as they are (default)
 * @param stop        optional text that ends generation early (not included in the result)
 * @returns only the newly generated characters, not `context`
 */
export function generate(
  model: Model,
  context: string,
  maxChars: number,
  temperature = 1.0,
  stop?: string,
): string {
  let output = context;
  for (let i = 0; i < maxChars; i++) {
    const windowText = output.slice(-model.config.blockSize);
    const contextIds = Array.from(windowText).map(
      (c) => model.charToId.get(c) ?? 0,
    );
    const { probs } = forward(model.weights, contextIds, temperature);
    output += model.idToChar.get(sample(probs))!;

    const generated = output.slice(context.length);
    if (stop && generated.endsWith(stop)) {
      return generated.slice(0, -stop.length);
    }
  }
  return output.slice(context.length);
}

// ==========================================
// Loading saved weights
// ==========================================

/**
 * Loads a trained model from disk and builds the character ↔ ID lookups.
 *
 * Throws an error telling you to retrain if the file was saved by an older
 * model version, because those weights would produce nonsense with the
 * current forward pass.
 *
 * @param path weights file to read (default "model-weights.json",
 *             relative to the folder you run the script from)
 * @returns the model, ready for generate()
 */
export function loadModel(path = WEIGHTS_FILE): Model {
  const saved: SavedModel = JSON.parse(fs.readFileSync(path, "utf-8"));
  if (saved.config?.version !== MODEL_VERSION) {
    throw new Error(
      `${path} was saved by an older model version (expected v${MODEL_VERSION}). ` +
        `Run "npm run train" to retrain.`,
    );
  }
  return {
    ...saved,
    charToId: new Map(saved.vocab.map((c, i) => [c, i])),
    idToChar: new Map(saved.vocab.map((c, i) => [i, c])),
  };
}
