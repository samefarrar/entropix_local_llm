from mlx_lm.utils import make_kv_caches
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple, Union, Dict
from mlx_attention_sampler import SamplerConfig
import csv
from pathlib import Path

LN_2 = 0.69314718056  # ln(2)

# ANSI escape sequences for text formatting
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_ITALIC = "\033[3m"
ANSI_DIM = "\033[2m"

# Catppuccin colour palette
COLOURS = {
    "rosewater": (244, 194, 193),
    "flamingo": (242, 150, 160),
    "pink": (245, 140, 173),
    "mauve": (203, 166, 247),
    "red": (243, 139, 168),
    "maroon": (235, 160, 172),
    "peach": (250, 179, 135),
    "yellow": (249, 226, 175),
    "green": (166, 227, 161),
    "teal": (148, 226, 213),
    "sky": (137, 220, 235),
    "sapphire": (116, 199, 236),
    "blue": (137, 180, 250),
    "lavender": (180, 190, 254),
    "text": (205, 214, 244),
    "subtext1": (186, 194, 222),
    "subtext0": (166, 173, 200),
    "overlay2": (147, 153, 178),
    "overlay1": (127, 132, 156),
    "overlay0": (108, 112, 134),
    "surface2": (88, 91, 112),
    "surface1": (69, 71, 90),
    "surface0": (49, 50, 68),
    "base": (30, 30, 46),
    "mantle": (24, 24, 37),
    "crust": (17, 17, 27)
}

def blend_colours(colour1: Tuple[int, int, int], colour2: Tuple[int, int, int], weight: float = 0.5) -> Tuple[int, int, int]:
    # Use a power function to emphasize brighter colours
    emphasis = 2.0
    w1 = weight ** (1/emphasis)
    w2 = (1 - weight) ** (1/emphasis)

    blended = tuple(int(((c1/255)**emphasis * w1 + (c2/255)**emphasis * w2) ** (1/emphasis) * 255)
                    for c1, c2 in zip(colour1, colour2))

    # Ensure the result is within valid RGB range
    blended = tuple(max(0, min(255, c)) for c in blended)

    #print(f"Debug: Blend result: {blended} (colour1: {colour1}, colour2: {colour2}, weight: {weight})", flush=True)
    return blended

def get_colour_for_metric(metrics: Dict[str, float], config) -> Tuple[Tuple[int, int, int], str]:
    """Get colour and formatting for metrics based on their values and thresholds."""
    ent = metrics["logits_entropy"]
    vent = metrics["logits_varentropy"]
    attn_ent = metrics["attention_entropy"]
    attn_vent = metrics["attention_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    colour = COLOURS["text"]  # Start with default text colour
    formatting = ""

    # Logits Entropy
    if ent < config.low_logits_entropy_threshold:
        colour = blend_colours(colour, COLOURS["blue"], 0.7)
    elif ent < config.medium_logits_entropy_threshold:
        colour = blend_colours(colour, COLOURS["sky"], 0.7)
    elif ent < config.high_logits_entropy_threshold:
        colour = blend_colours(colour, COLOURS["sapphire"], 0.7)
    else:
        colour = blend_colours(colour, COLOURS["lavender"], 0.7)

    # Logits Varentropy
    if vent < config.low_logits_varentropy_threshold:
        colour = blend_colours(colour, COLOURS["green"], 0.3)
    elif vent < config.medium_logits_varentropy_threshold:
        colour = blend_colours(colour, COLOURS["teal"], 0.3)
    elif vent < config.high_logits_varentropy_threshold:
        colour = blend_colours(colour, COLOURS["yellow"], 0.3)
    else:
        colour = blend_colours(colour, COLOURS["peach"], 0.3)

    # Attention Entropy
    if attn_ent < config.low_attention_entropy_threshold:
        formatting += ANSI_BOLD
    elif attn_ent > config.high_attention_entropy_threshold:
        formatting += ANSI_ITALIC

    # Attention Varentropy
    if attn_vent < config.low_attention_varentropy_threshold:
        colour = blend_colours(colour, COLOURS["rosewater"], 0.2)
    elif attn_vent < config.medium_attention_varentropy_threshold:
        colour = blend_colours(colour, COLOURS["flamingo"], 0.2)
    elif attn_vent < config.high_attention_varentropy_threshold:
        colour = blend_colours(colour, COLOURS["pink"], 0.2)
    else:
        colour = blend_colours(colour, COLOURS["mauve"], 0.2)

    #print(f"Pre-agreement colour: {colour}")

    # Agreement
    if agreement < config.low_agreement_threshold:
        formatting += ANSI_DIM
    elif agreement > config.high_agreement_threshold:
        colour = blend_colours(colour, COLOURS["red"], 0.2)

    # print(f"Pre-interaction strength colour: {colour}")

    # Interaction Strength
    if interaction_strength < config.low_interaction_strength_threshold:
        colour = blend_colours(colour, COLOURS["surface2"], 0.1)
    elif interaction_strength < config.medium_interaction_strength_threshold:
        colour = blend_colours(colour, COLOURS["surface1"], 0.1)
    elif interaction_strength < config.high_interaction_strength_threshold:
        colour = blend_colours(colour, COLOURS["surface0"], 0.1)
    else:
        colour = blend_colours(colour, COLOURS["base"], 0.1)

    # print(f"Final colour: {colour}")
    return colour, formatting

@mx.compile
def calculate_varentropy_logsoftmax(
    logits: mx.array, axis: int = -1
) -> tuple[mx.array, mx.array]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = mx.softmax(logits, axis=axis).log()
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = mx.sum(probs * (log_probs / LN_2 + entropy[..., None]) ** 2, axis=axis)
    return entropy, varentropy

@mx.compile
def calculate_metrics(logits: mx.array, attention_scores: mx.array) -> Dict[str, mx.array]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = mx.softmax(attention_scores, axis=-1)
    attention_entropy = -mx.sum(attention_probs * mx.log2(mx.clip(attention_probs, 1e-10, 1.0)), axis = -1)
    attention_varentropy = mx.var(attention_entropy, axis = 1)

    mean_attention = mx.mean(attention_probs, axis = 1)
    agreement = mx.mean(mx.abs(attention_probs - mean_attention[:, None, :]), axis = (1, 2))

    interaction_strength = mx.mean(mx.abs(attention_scores), axis = (1, 2, 3))

    return {
        "logits_entropy": mx.mean(entropy),
        "logits_varentropy": mx.mean(varentropy),
        "attention_entropy": mx.mean(attention_entropy),
        "attention_varentropy": mx.mean(attention_varentropy),
        "agreement": mx.mean(agreement),
        "interaction_strength": interaction_strength
    }

def _sample(logits: mx.array,
    temperature=0.666,
    top_p=0.9,
    top_k: int = 27,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 2) -> mx.array:
    batch_size = logits.shape[0]
    logit = logits[:, -1] / temperature  # (batch_size, vocab_size)

    # Calculate probabilities by softmaxing the temparature-scaled logits
    probs = mx.softmax(logit, axis=-1)

    # Sort probabilities in descending order
    # This should then look like
    sorted_indices = mx.argsort(-probs, axis=-1) # e.g. (bsz x [3, 1280, 1, 0, 2, ...])
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1) # e.g. (bsz x [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Apply min_p sampling
    if min_p > 0:
        top_prob = sorted_probs[..., 0] # Highest probability e.g. (bsz x[0.9])
        scaled_min_p = min_p * top_prob # e.g. 0.9 * 0.1 = 0.09, (bsz x[0.09])
        min_p_mask = sorted_probs > scaled_min_p[..., None] # e.g. (bsz * [True, False, False, False, False, ...])
        min_p_mask[..., :min_tokens_to_keep] = True # Keep at least min_tokens_to_keep tokens, e.g. (bsz * [True, True, True, False, False, ...])
        sorted_probs = mx.where(min_p_mask, sorted_probs, 0.0) # e.g. (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...])

    # Apply top_p (nucleus) sampling
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1, inclusive = False) # e.g. (bsz * [0.9, 0.95, 0.97, 0.98, 0.99, ...]
    # or, if min_p is applied, (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...]
    top_p_mask = cumulative_probs <= top_p # e.g. (bsz * [True, True, True, True, True, ...]
    # or, if min_p is applied, (bsz * [True, False, False, False, False, ...]
    top_p_mask[..., :min_tokens_to_keep] = True # Keep at least min_tokens_to_keep tokens, e.g. (bsz * [True, True, True, False, False, ...])
    sorted_probs = mx.where(top_p_mask, sorted_probs, 0.0) # e.g. (bsz * [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Optionally apply top_k sampling
    sorted_probs[..., top_k:] = 0.0 # e.g. (bsz * [0.9, 0.05, 0.0, 0.0, 0.0, ...])

    # Sample token
    sorted_token = mx.random.categorical(mx.log(sorted_probs))[..., None] # e.g. (bsz * [1390, 3, 2791, 1381, 12476, ...])
    token = mx.take_along_axis(sorted_indices, sorted_token, axis=-1) # e.g. [3,] in shape (batch_size,)
    return token

@mx.compile
def score_sample(
    sample: mx.array,
    logits: mx.array,
    ent: float,
    attention_entropy: float,
    vent: float,
    attention_varentropy: float,
    agreement: float,
    interaction_strength: float,
    high_logits_entropy_threshold: float,
    adaptive_score_logits_entropy_coefficient: float,
    high_attention_entropy_threshold: float,
    adaptive_score_attention_entropy_coefficient: float,
    high_logits_varentropy_threshold: float,
    adaptive_score_logits_varentropy_coefficient: float,
    high_attention_varentropy_threshold: float,
    adaptive_score_attention_varentropy_coefficient: float,
    high_agreement_threshold: float,
    adaptive_score_agreement_coefficient: float,
    high_interaction_strength_threshold: float,
    adaptive_score_interaction_strength_coefficient: float
):
    batch_size, seq_length = sample.shape
    vocab_size = logits.shape[-1]
    # Create one-hot encoding
    one_hot = mx.zeros((batch_size, seq_length, vocab_size))
    one_hot[mx.arange(batch_size)[:, None], mx.arange(seq_length)[None, :], sample] = 1
    # Calculate log probability
    log_probs = mx.sum(mx.softmax(logits[:, -1], axis=-1).log()[:, None, :] * one_hot, axis=(1, 2))
    # Calculate confidence score
    confidence_scores = (
        (1 - ent / high_logits_entropy_threshold) * adaptive_score_logits_entropy_coefficient +
        (1 - attention_entropy / high_attention_entropy_threshold) * adaptive_score_attention_entropy_coefficient +
        (1 - vent / high_logits_varentropy_threshold) * adaptive_score_logits_varentropy_coefficient +
        (1 - attention_varentropy / high_attention_varentropy_threshold) * adaptive_score_attention_varentropy_coefficient +
        (agreement / high_agreement_threshold) * adaptive_score_agreement_coefficient +
        (interaction_strength / high_interaction_strength_threshold) * adaptive_score_interaction_strength_coefficient
    )
    return log_probs + confidence_scores

def sample(
    gen_tokens: mx.array, logits: mx.array, scores: mx.array, cfg: SamplerConfig, clarifying_question_token: int = 2564
) -> Tuple[mx.array, Tuple[Tuple[int,int,int], str]]:
    metrics = calculate_metrics(logits, scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attention_entropy, attention_varentropy = metrics["attention_entropy"], metrics["attention_varentropy"]
    agreement, interaction_strength = metrics["agreement"], metrics["interaction_strength"]
    colour = get_colour_for_metric(metrics, cfg)
    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (ent < cfg.low_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold):# and
        # attention_entropy < cfg.low_attention_entropy_threshold and
        # attention_varentropy < cfg.low_attention_varentropy_threshold and
        # agreement < cfg.low_agreement_threshold and
        # interaction_strength < cfg.low_interaction_strength_threshold):
        #print("🌊", flush = True, end = "")
        return mx.argmax(logits[:, -1], axis=-1, keepdims=True), colour
    # # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (ent > cfg.high_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold): # and
          # attention_entropy < cfg.low_attention_entropy_threshold and
          # attention_varentropy < cfg.low_attention_varentropy_threshold and
          # agreement < cfg.low_agreement_threshold and
          # interaction_strength < cfg.low_interaction_strength_threshold):
        #print("ε", flush = True, end = "")
        # Insert a clarifying question token if not already present
        if not mx.any(mx.equal(gen_tokens[:, -1], clarifying_question_token).any()):
            return (mx.array(
                [[clarifying_question_token]]
            ), colour)  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attention_entropy  # Increase temperature based on attention entropy
            return (_sample(logits,
                temperature=min(1.5, cfg.temperature * temp_adj),
                top_p = cfg.top_p,
                top_k = cfg.top_k,
                min_p = cfg.min_probability),
            colour)
    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif (ent < cfg.high_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold):# and
          # attention_entropy < cfg.low_attention_entropy_threshold and
          # attention_varentropy > cfg.high_attention_varentropy_threshold and
          # agreement < cfg.low_agreement_threshold and
          # interaction_strength > cfg.low_interaction_strength_threshold):
        #print("Φ", flush = True, end = "")
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        # top_k_values, top_k_indices = mx.top_k(logits[:, -1], k=top_k)
        # return top_k_indices
        temp_adj = cfg.low_entropy_interaction_strength_offset + cfg.low_entropy_interaction_strength_coefficient * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement)))) # Increase top_k when agreement is low
        return (_sample(logits,
            temperature=min(1,5, cfg.temperature * temp_adj),
            top_p = cfg.top_p,
            top_k = top_k_adj,
            min_p = cfg.min_probability),
        colour)
    # High Entropy, High Varentropy: "resampling in the mist"
    elif (ent > cfg.medium_logits_entropy_threshold and
        vent > cfg.high_logits_varentropy_threshold): # and
          # attention_entropy > cfg.high_attention_entropy_threshold and
          # attention_varentropy > cfg.high_attention_varentropy_threshold and
          # agreement > cfg.high_agreement_threshold and
          # interaction_strength > cfg.high_interaction_strength_threshold):
        #print("Ω", flush = True, end = "")
        # Use high temperature and min_p sampling
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attention_varentropy  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attention_entropy)  # Decrease top_p when attention entropy is high
        return (_sample(logits,
            temperature=max(2.0,
                cfg.temperature * temp_adj),
            top_p = top_p_adj,
            top_k = cfg.top_k,
            min_p = cfg.min_probability),
        colour)
    # Middle ground: smooth transition
    else:
        #print("Ψ", flush = True, end = "")
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attention_uncertainty = metrics["attention_entropy"] + metrics["attention_varentropy"]
        temperature = cfg.temperature * (
            1 +
            cfg.adaptive_temperature_logits_coefficient * ent +
            cfg.adaptive_temperature_attention_coefficient * attention_entropy -
            cfg.adaptive_temperature_agreement_coefficient * agreement
        )
        top_p = mx.clip(
            cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attention_varentropy),
            0.1,
            1.0)
        top_k = int(
            mx.clip(
                mx.round(cfg.top_k * (1 + cfg.adaptive_top_k_interaction_coefficient * interaction_strength.item() - cfg.adaptive_top_k_agreement_coefficient * metrics['agreement'].item())),
                a_min = 1,
                a_max = 100
            )
        )
        min_p = mx.clip(
            cfg.min_probability * (1 - cfg.adaptive_min_p_coefficient * vent),
            0.01,
            0.5
        )

        # Sample from the logits
        samples = _sample(mx.repeat(logits, cfg.number_of_adaptive_samples, axis = 0), temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)

        sample_scores = score_sample(
            samples,
            mx.repeat(logits, cfg.number_of_adaptive_samples, axis=0),
            metrics["logits_entropy"],
            metrics["attention_entropy"],
            metrics["logits_varentropy"],
            metrics["attention_varentropy"],
            metrics["agreement"],
            metrics["interaction_strength"],
            cfg.high_logits_entropy_threshold,
            cfg.adaptive_score_logits_entropy_coefficient,
            cfg.high_attention_entropy_threshold,
            cfg.adaptive_score_attention_entropy_coefficient,
            cfg.high_logits_varentropy_threshold,
            cfg.adaptive_score_logits_varentropy_coefficient,
            cfg.high_attention_varentropy_threshold,
            cfg.adaptive_score_attention_varentropy_coefficient,
            cfg.high_agreement_threshold,
            cfg.adaptive_score_agreement_coefficient,
            cfg.high_interaction_strength_threshold,
            cfg.adaptive_score_interaction_strength_coefficient
        )
        best_sample_idx = mx.argmax(sample_scores)
        return samples[best_sample_idx][None], colour
