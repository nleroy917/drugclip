
from typing import Dict, Optional

from transformers import PretrainedConfig

from unimol import UniMolConfig


class DrugCLIPConfig(PretrainedConfig):
    """
    Configuration for DrugCLIP model.
    """

    model_type = "drugclip"

    def __init__(
        self,
        mol_vocab_size: int = 31,
        pocket_vocab_size: int = 10,
        hidden_size: int = 512,
        num_hidden_layers: int = 15,
        num_attention_heads: int = 64,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        activation_dropout: float = 0.0,
        max_position_embeddings: int = 512,
        gbf_kernels: int = 128,
        projection_dim: int = 128,
        logit_scale_init: float = 2.6592,  # ln(14)
        mol_dictionary: Optional[Dict[str, int]] = None,
        pocket_dictionary: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mol_vocab_size = mol_vocab_size
        self.pocket_vocab_size = pocket_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.gbf_kernels = gbf_kernels
        self.projection_dim = projection_dim
        self.logit_scale_init = logit_scale_init
        self.mol_dictionary = mol_dictionary or {}
        self.pocket_dictionary = pocket_dictionary or {}

    def get_mol_config(self) -> UniMolConfig:
        """
        Get configuration for molecule encoder.
        """
        return UniMolConfig(
            vocab_size=self.mol_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            activation_dropout=self.activation_dropout,
            max_position_embeddings=self.max_position_embeddings,
            gbf_kernels=self.gbf_kernels,
        )

    def get_pocket_config(self) -> UniMolConfig:
        """
        Get configuration for pocket encoder.
        """
        return UniMolConfig(
            vocab_size=self.pocket_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            activation_dropout=self.activation_dropout,
            max_position_embeddings=self.max_position_embeddings,
            gbf_kernels=self.gbf_kernels,
        )