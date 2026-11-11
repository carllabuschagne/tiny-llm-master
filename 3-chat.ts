import * as readline from "readline";
import { generate, loadModel } from "./model.ts";

/** The trained model, loaded from model-weights.json. */
const model = loadModel();

/** Low temperature = more confident, less random replies. */
const temperature = 0.25;

/** The most characters a single reply can be. */
const maxReplyChars = 100;

/**
 * Generates the model's reply to what the user typed.
 *
 *  1. Cleans the prompt: lowercases it and removes characters the model
 *     doesn't know. An empty result falls back to "hi".
 *  2. Wraps it in the same format as data.txt:
 *       user: <prompt>
 *       bot: <- the model continues from here
 *     Only this one exchange is sent. Training windows never reach back past
 *     the start of an exchange either, so the model sees exactly what it
 *     was trained on, with every character in the same position.
 *  3. Generates until the end of the bot's line (a newline), so the model
 *     doesn't go on to invent the user's next message.
 *
 * @param prompt what the user typed
 * @returns the bot's reply
 */
function generateReply(prompt: string): string {
  let cleanPrompt = Array.from(prompt.toLowerCase())
    .filter((c) => model.charToId.has(c))
    .join("")
    .trim();

  if (cleanPrompt.length === 0) cleanPrompt = "hi";

  const context = `user: ${cleanPrompt}\nbot: `;
  return generate(model, context, maxReplyChars, temperature, "\n").trim();
}

/** Terminal input/output used to read prompts and print replies. */
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log("Mini-GPT Loaded. Type a message or 'exit' to quit.\n");

/**
 * The chat loop: asks for a message, prints the bot's reply, then calls
 * itself to ask again. Typing "exit" closes the terminal input and ends
 * the program.
 */
function promptUser() {
  rl.question("You > ", (userInput) => {
    if (userInput.trim().toLowerCase() === "exit") {
      rl.close();
      return;
    }

    console.log(`Bot > ${generateReply(userInput)}\n`);
    promptUser();
  });
}

promptUser();
