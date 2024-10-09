from pydantic import BaseModel
class SamplerConfig(BaseModel):
    """
    Configuration for the sampling strategy, including threshold values for various metrics
    and adaptive sampling parameters.
    """

    # Sampling Hyperparameters
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_probability: float = 0.03  # Minimum probability threshold for token selection

    # Logits Entropy Thresholds
    low_logits_entropy_threshold: float = 0.1
    medium_logits_entropy_threshold: float = 3.0
    high_logits_entropy_threshold: float = 5.0

    # Logits Varentropy Thresholds
    low_logits_varentropy_threshold: float = 0.1
    medium_logits_varentropy_threshold: float = 3.0
    high_logits_varentropy_threshold: float = 5.0

    # Attention Entropy Thresholds
    low_attention_entropy_threshold: float = 3.0
    medium_attention_entropy_threshold: float = 3.8
    high_attention_entropy_threshold: float = 4.0

    # Attention Varentropy Thresholds
    low_attention_varentropy_threshold: float = 0.9
    medium_attention_varentropy_threshold: float = 1.5
    high_attention_varentropy_threshold: float = 1.9

    # Agreement Thresholds
    low_agreement_threshold: float = 0.00065
    medium_agreement_threshold: float = 0.0010
    high_agreement_threshold: float = 0.0020

    # Interaction Strength Thresholds
    low_interaction_strength_threshold: float = 5.53
    medium_interaction_strength_threshold: float = 6.59
    high_interaction_strength_threshold: float = 6.87


    # Offsets and Coefficients for Adjusting Sampling Parameters
    high_entropy_attention_offset: float = 1.3
    high_entropy_attention_coefficient: float = 0.2 / 0.29

    low_entropy_interaction_strength_offset: float = 1.2
    low_entropy_interaction_strength_coefficient: float = 0.3 / 27.2

    high_entropy_varentropy_attention_offset: float = 2.0
    high_entropy_varentropy_attention_coefficient: float = 0.5 / 310.0

    # Adaptive Sampling Parameters
    number_of_adaptive_samples: int = 5

    adaptive_temperature_logits_coefficient: float = 0.3
    adaptive_temperature_attention_coefficient: float = 0.2 / 0.29
    adaptive_temperature_agreement_coefficient: float = 0.2 / 248.0
    adaptive_top_p_coefficient: float = 0.1 / 310.0
    adaptive_top_k_interaction_coefficient: float = 0.3 / 27.10
    adaptive_top_k_agreement_coefficient: float = 0.2 / 248.0
    adaptive_min_p_coefficient: float = 0.5
    adaptive_score_logits_entropy_coefficient: float = 0.1
    adaptive_score_attention_entropy_coefficient: float = 0.2 / 0.31
    adaptive_score_logits_varentropy_coefficient: float = 0.3
    adaptive_score_attention_varentropy_coefficient: float = 0.4 / 310.0
    adaptive_score_agreement_coefficient: float = 0.5 / 248.0
    adaptive_score_interaction_strength_coefficient: float = 0.6 / 27.10
