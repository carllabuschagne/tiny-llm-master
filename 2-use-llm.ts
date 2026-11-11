import { generate, loadModel } from "./model.ts";

// 1. LOAD THE SAVED BRAIN
const model = loadModel();

// 2. GENERATE (temperature 1.0 = sample straight from the learned probabilities)
// Starts a conversation in the data.txt format and lets the model write the rest,
// both the bot's replies and the user's next messages.
const prompt = "user: hi\nbot: ";
console.log(prompt + generate(model, prompt, 120, 1.0));
