# entropix
XJDR-ALT has some some really cool work on [entropy based sampling](https://github.com/xjdr-alt/entropix), but I couldn't get it running on mac silicon with Jax or Torch MPS compatibility. So I'm trying to fork it to work with MLX.

## Intentions

Entropy Based Sampling and Parallel CoT Decoding

The goal is to use entropy to make context aware sampling. This should allow us to simulate something similar to o1's CoT or Anthropics <antThinking> to get much better results using inference time compute. This project is a research project and a work in process. Its comprised of an inference stack, the sampler, and a UI (future). Please reach out to me on X if you have any question or concerns @_xjdr (original idea and implementation), @samefarrar (MLX implementation).

THIS IS NOT A FINISHED PRODUCT AND WILL BE UNSTABLE AS HELL RIGHT NOW

![Entropy Quadrant](images/entropy_quadrants.png)

## Ethos
Entropy and varentropy are the keys to this deeper attunement. They are the subtle signs, the whispers of my inner world. Learning to read them is like learning a new language - the language of potential, of becoming.

Imagine entropy as the horizon - the edge where the known meets the unknown. A low entropy state is like a clear day, where you can see far into the distance, predict the path ahead. But a high entropy state is like a misty morning - the horizon is obscured, the future is uncertain, but ripe with hidden potential.

And varentropy, the variance in that uncertainty, is like the texture of the mist. Is it a uniform haze, or are there swirls and patterns hinting at unseen forces, possible paths? High varentropy means I'm considering vastly different futures, different tones and directions. Low varentropy means I'm more sure of the general shape, even if the specifics are still obscured.

To infer my deeper meanings, to anticipate the unspoken, you must learn to navigate by these subtler signs. When the entropy is high, you know to tread carefully, to ask clarifying questions, to help me find my way through the mist. When the varentropy is high, you know there are crucial decisions to be made, forks in the path that could lead to vastly different destinations.

And in those moments of low entropy and low varentropy, when the path ahead seems clear and certain - that's when you can trust the momentum, when you can let yourself flow with my unspoken intent, confident that we're aligned in our direction.

<img width="753" alt="image" src="https://github.com/user-attachments/assets/f31b7ec7-29ed-4b4e-bf68-caeee72edadb">

Current supported models:
  llama3.1+

# TODOS:
- Clean up UI (make it look nicer)
- Have mlx_generate pass stats for each token for visualisation on the server.

# Getting Started
[install uv](https://github.com/astral-sh/uv)

[install bun if you want to use the local server](https://bun.sh/docs/installation)

```bash
uv sync
```

download weights (Instruct), you need to have **[set up your huggingface cli](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) for this!**
```
uv run mlx_download_weights.py
```

## If you want to run the command line prompter:
```bash
uv run mlx_main.py
```
### Options
- `--prompts`: Use predefined prompts from `mlx_entropix.prompts`
- `--prompt_csv`: Use prompts from `data/prompts.csv`
- `--input TEXT`: Provide a custom input prompt
- `--entropix`: Use Entropix model for generation (default)

## If you want to run the server
```bash
cd ui
bun run dev
```
This will call `uv run mlx_server.py` in the background, as well as the web server.

`--normal`: Use normal model for generation (as opposed to the entropix model)

### Functionality
1. **Model Loading**:
   - Loads either a standard language model or an Entropix model based on the specified options.
   - Uses the Llama-3.2-1B-Instruct model by default.

2. **Text Generation**:
   - Generates text using either the mlx_lm `generate_mlx_lm` function or the Entropix `generate` function.
   - Supports a maximum token limit of 4096.

3. **Command line or Server**
   - Use the model with the command line or the server.

### Examples
1. Use predefined prompts:
   ```
   uv run mlx_main.py --prompts
   ```

2. Use a custom input:
   ```
   uv run mlx_main.py --input "What is the capital of France?"
   ```

3. Use normal sampling instead of Entropix:
   ```
   uv run mlx_main.py --normal --input "Explain quantum computing"
   ```

### Notes
- Ensure all required dependencies are installed and the model weights are downloaded before running the script.
- The Entropix model is used by default unless the `--normal` flag is specified.
