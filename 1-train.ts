import * as fs from "fs";
import {
  forward,
  layerNormBackward,
  matVecMulT,
  MODEL_VERSION,
  WEIGHTS_FILE,
  type Matrix,
  type SavedModel,
  type Weights,
} from "./model.ts";

// ==========================================
// 1. TRAINING DATA & TOKENIZER
// ==========================================

/** File the training conversations are read from. */
const DATA_FILE = "data.txt";

/**
 * The text the model learns from: the conversations in data.txt.
 *
 * Windows line endings (\r\n) are turned into plain \n, and everything is
 * lowercased to keep the vocabulary small (the chat lowercases prompts too).
 * Edit data.txt and retrain to change what the bot says.
 */
const trainingText = fs
  .readFileSync(DATA_FILE, "utf-8")
  .replace(/\r\n/g, "\n")
  .toLowerCase()
  .trim();

/** Every unique character in the training text, sorted. A character's index is its ID. */
const vocab = Array.from(new Set(trainingText)).sort();
/** How many unique characters the model knows. */
const vocabSize = vocab.length;
/** Tokenizer: character → ID. */
const charToId = new Map<string, number>(vocab.map((c, i) => [c, i]));

// ==========================================
// 2. SETTINGS (hyperparameters)
// ==========================================

/**
 * Context window: how many previous characters the model looks at.
 * 32 is enough to still see a short question ("user: what is your name?\nbot: ")
 * while writing the answer.
 */
const blockSize = 32;
/** Length of each character's vector. Bigger = more capacity, slower training. */
const embedDim = 32;
/** Width of the MLP's middle layer. 4 × embedDim is the standard GPT choice. */
const mlpDim = 4 * embedDim;
/**
 * Learning rate schedule: how big each weight nudge is.
 * It starts at startLearningRate and is lowered steadily to endLearningRate
 * by the last epoch. Big steps learn fast early on; small steps late in
 * training avoid the sudden spikes that can undo what was learned.
 * Too high: the loss climbs or becomes NaN. Too low: slow learning.
 */
const startLearningRate = 0.01;
const endLearningRate = 0.001;
/** The learning rate for the current epoch (updated at the start of each epoch). */
let learningRate = startLearningRate;
/**
 * Gradient clipping: no single gradient value can be bigger than this (either sign).
 * A safety net so one unusually large gradient can't throw a weight far off and
 * blow up training.
 */
const gradClip = 1.0;
/** Print the loss every this many epochs. */
const logEvery = 50;
/** How many times to read through the whole training text. */
const epochs = 1000;

/**
 * Creates a matrix filled with small random numbers between -0.1 and 0.1.
 *
 * Weights start random so the model begins knowing nothing; training then
 * shapes them. They're kept small so the first predictions are close to
 * even guesses rather than wildly confident wrong ones.
 *
 * @param rows number of rows
 * @param cols number of columns
 * @returns a new rows × cols matrix
 */
function randMatrix(rows: number, cols: number): Matrix {
  const mat: Matrix = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      row.push((Math.random() - 0.5) * 0.2);
    }
    mat.push(row);
  }
  return mat;
}

/** All of the model's weights, randomly initialised. Training updates these in place. */
const weights: Weights = {
  tokenEmbed: randMatrix(vocabSize, embedDim),
  posEmbed: randMatrix(blockSize, embedDim),
  Wq: randMatrix(embedDim, embedDim),
  Wk: randMatrix(embedDim, embedDim),
  Wv: randMatrix(embedDim, embedDim),
  W1: randMatrix(embedDim, mlpDim),
  W2: randMatrix(mlpDim, embedDim),
  Wout: randMatrix(embedDim, vocabSize),
};
const { tokenEmbed, posEmbed, Wq, Wk, Wv, W1, W2, Wout } = weights;

/**
 * Limits a gradient value to the range -gradClip..+gradClip.
 *
 * Example (gradClip = 1): clip(0.4) → 0.4, clip(25) → 1, clip(-3) → -1
 *
 * @param g a gradient value
 * @returns the clipped value
 */
function clip(g: number): number {
  return Math.max(-gradClip, Math.min(gradClip, g));
}

/**
 * Applies one gradient-descent step to a weight matrix.
 *
 * For a layer computed as `out = input · mat`, the gradient for each weight
 * mat[i][j] is input[i] × gradOut[j]. Every weight is moved a small step
 * (learningRate) in the opposite direction, to reduce the error.
 * Each gradient is clipped first (see gradClip).
 *
 * @param mat     the weight matrix to update (changed in place)
 * @param input   the vector that was fed into the layer
 * @param gradOut the gradient (error) on the layer's output
 */
function stepOuter(mat: Matrix, input: number[], gradOut: number[]) {
  for (let i = 0; i < input.length; i++) {
    for (let j = 0; j < gradOut.length; j++) {
      mat[i][j] -= learningRate * clip(input[i] * gradOut[j]);
    }
  }
}

/**
 * Applies one gradient-descent step to a single vector, such as one
 * character's embedding row: vec[d] -= learningRate × grad[d].
 * Each gradient is clipped first (see gradClip).
 *
 * @param vec  the vector to update (changed in place)
 * @param grad its gradient, the same length as vec
 */
function stepVec(vec: number[], grad: number[]) {
  for (let d = 0; d < vec.length; d++) {
    vec[d] -= learningRate * clip(grad[d]);
  }
}

// ==========================================
// 3. TRAINING LOOP
// ==========================================
// Each epoch reads through the whole text. At every position the model
// predicts the next character from the previous (up to blockSize)
// characters, measures how wrong it was, and nudges every weight to do
// a bit better next time.

/**
 * Shuffles an array in place into a random order (Fisher-Yates shuffle).
 *
 * Used to visit the training positions in a different order every epoch.
 * Going through the text in the same order every time makes the model keep
 * "chasing" whatever it saw last; a random order gives steadier learning.
 *
 * @param arr the array to shuffle (changed in place)
 */
function shuffle(arr: number[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

/** Every position in the text that has a character to predict (1 .. length-1). */
const positions = Array.from(
  { length: trainingText.length - 1 },
  (_, k) => k + 1,
);

/**
 * For every position i: where its exchange begins, i.e. the index of the most
 * recent "user:" line that starts BEFORE i.
 *
 * Training windows are cut off there, so the model only ever sees the current
 * exchange and never the end of the previous answer. That matches the chat,
 * which sends just "user: <message>\nbot: ", so every character sits in the
 * same position in training and in the chat.
 */
const exchangeStart: number[] = [];
{
  let lastStart = 0;
  for (let i = 0; i < trainingText.length; i++) {
    exchangeStart.push(lastStart);
    const lineStart = i === 0 || trainingText[i - 1] === "\n";
    if (lineStart && trainingText.startsWith("user:", i)) lastStart = i;
  }
}

console.log("Training Mini-GPT...");

/** The lowest average loss seen so far, which epoch it was, and a copy of the weights at that point. */
let bestLoss = Infinity;
let bestEpoch = -1;
let bestWeights: Weights = structuredClone(weights);

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;
  let steps = 0;

  // Lower the learning rate in a straight line from start to end over all epochs
  learningRate =
    startLearningRate +
    (endLearningRate - startLearningRate) * (epoch / Math.max(1, epochs - 1));

  shuffle(positions);

  for (const i of positions) {
    // The characters before position i are the input, and the character at i is the answer.
    // The window holds up to blockSize characters, but never reaches back past the
    // start of the current exchange.
    const start = Math.max(exchangeStart[i], i - blockSize);
    const contextIds = Array.from(trainingText.slice(start, i)).map(
      (c) => charToId.get(c)!,
    );
    const targetId = charToId.get(trainingText[i])!;

    // ---- FORWARD PASS: make a prediction ----
    const {
      X,
      N,
      nInvStd,
      Q,
      K,
      V,
      attnWeights,
      M,
      mInvStd,
      mlpPre,
      mlpAct,
      H,
      hInvStd,
      probs,
    } = forward(weights, contextIds);

    // Loss: how surprised the model was by the right answer (0 = certain and correct)
    totalLoss += -Math.log(probs[targetId] + 1e-9);
    steps++;

    const T = contextIds.length;
    const lastIdx = T - 1;
    const scale = Math.sqrt(embedDim);

    // ---- BACKWARD PASS: compute every gradient first, using the current weights ----
    // Each step below undoes one step of forward(), in reverse order, working out
    // how much each value contributed to the error.

    // Loss -> logits
    const dLogits = probs.map((p, idx) => p - (idx === targetId ? 1 : 0));

    // logits = H · Wout  ->  gradient on H
    const dH = matVecMulT(Wout, dLogits);

    // H = layerNorm(hidden2)  ->  gradient on hidden2
    const dHidden2 = layerNormBackward(H, hInvStd, dH);

    // hidden2 = hidden + mlpOut  ->  both receive dHidden2 (the MLP's residual path)
    const dMlpOut = dHidden2;

    // mlpOut = mlpAct · W2  ->  gradient on mlpAct
    const dMlpAct = matVecMulT(W2, dMlpOut);

    // mlpAct = relu(mlpPre)  ->  the gradient only passes where the value was positive
    const dMlpPre = dMlpAct.map((g, k) => (mlpPre[k] > 0 ? g : 0));

    // mlpPre = M · W1  ->  gradient on M
    const dM = matVecMulT(W1, dMlpPre);

    // M = layerNorm(hidden)  ->  gradient on hidden: through the MLP plus the residual
    const fromMlp = layerNormBackward(M, mInvStd, dM);
    const dHidden = dHidden2.map((g, d) => g + fromMlp[d]);

    // hidden = X[last] + contextVec  ->  both receive dHidden (the residual path)
    const dContext = dHidden;
    const dX: Matrix = X.map(() => new Array(embedDim).fill(0));
    for (let d = 0; d < embedDim; d++) dX[lastIdx][d] += dHidden[d];

    // contextVec = sum_t attn[t] * V[t]
    const dV: Matrix = attnWeights.map((a) => dContext.map((g) => a * g));
    const dAttn = V.map((v) =>
      v.reduce((acc, vd, d) => acc + vd * dContext[d], 0),
    );

    // Softmax backward for the attention scores
    const sumDAttn = dAttn.reduce(
      (acc, val, t) => acc + val * attnWeights[t],
      0,
    );
    const dScore = attnWeights.map(
      (a, t) => (a * (dAttn[t] - sumDAttn)) / scale,
    );

    // score[t] = Q[last] · K[t]
    const dQ = new Array(embedDim).fill(0);
    const dK: Matrix = K.map(() => new Array(embedDim).fill(0));
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < embedDim; d++) {
        dQ[d] += dScore[t] * K[t][d];
        dK[t][d] = dScore[t] * Q[lastIdx][d];
      }
    }

    // Q/K/V = N · W  ->  gradient on each normalised position N[t]
    const dN: Matrix = N.map(() => new Array(embedDim).fill(0));
    for (let t = 0; t < T; t++) {
      const fromV = matVecMulT(Wv, dV[t]);
      const fromK = matVecMulT(Wk, dK[t]);
      for (let d = 0; d < embedDim; d++) dN[t][d] += fromV[d] + fromK[d];
    }
    const fromQ = matVecMulT(Wq, dQ);
    for (let d = 0; d < embedDim; d++) dN[lastIdx][d] += fromQ[d];

    // N[t] = layerNorm(X[t])  ->  gradient flows back into each position's input X[t]
    for (let t = 0; t < T; t++) {
      const fromN = layerNormBackward(N[t], nInvStd[t], dN[t]);
      for (let d = 0; d < embedDim; d++) dX[t][d] += fromN[d];
    }

    // ---- UPDATE: apply all gradients ----
    stepOuter(Wout, H, dLogits);
    stepOuter(W2, mlpAct, dMlpOut);
    stepOuter(W1, M, dMlpPre);
    stepOuter(Wq, N[lastIdx], dQ);
    for (let t = 0; t < T; t++) {
      stepOuter(Wv, N[t], dV[t]);
      stepOuter(Wk, N[t], dK[t]);
      // X[t] = tokenEmbed[id] + posEmbed[t], so both get the same gradient
      stepVec(tokenEmbed[contextIds[t]], dX[t]);
      stepVec(posEmbed[t], dX[t]);
    }
  }

  // NaN guard: if training has blown up, stop without saving, so a failed run
  // never overwrites good weights.
  if (!Number.isFinite(totalLoss)) {
    console.error(
      `Epoch ${epoch} | Loss became NaN: training blew up. ` +
        `Nothing was saved. Try a lower learningRate.`,
    );
    process.exit(1);
  }

  const avgLoss = totalLoss / steps;

  // Keep the best weights: if this is the lowest loss so far, save a copy.
  // (The loss is measured while the weights change during the epoch, so it's
  // a close estimate of how good the copy at the end of the epoch is.)
  if (avgLoss < bestLoss) {
    bestLoss = avgLoss;
    bestEpoch = epoch;
    bestWeights = structuredClone(weights);
  }

  //console.log(steps);

  // Report the average loss now and then. It should fall steadily from about 3.0
  //if (epoch % logEvery === 0 || epoch === epochs - 1) {
  console.log(
    `Epoch ${epoch} / ${epochs} | Error Loss: ${avgLoss.toFixed(4)} | ` +
      `Learning rate: ${learningRate.toFixed(4)}`,
  );
  //}
}

// ==========================================
// 4. SAVE THE TRAINED MODEL
// ==========================================

/**
 * Everything 2-use-llm.ts and 3-chat.ts need to run the model, in the
 * model-weights.json format. Uses the BEST weights, not the last epoch's,
 * so a late spike in the loss can't ruin the saved model.
 */
const savedModel: SavedModel = {
  config: { version: MODEL_VERSION, blockSize, embedDim, vocabSize },
  vocab,
  trainingText,
  weights: bestWeights,
};

fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(savedModel));
console.log(
  `Saved ${WEIGHTS_FILE} with the best weights, ` +
    `from epoch ${bestEpoch} (loss ${bestLoss.toFixed(4)}).`,
);
