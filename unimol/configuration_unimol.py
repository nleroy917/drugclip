from transformers import PretrainedConfig


class UniMolConfig(PretrainedConfig):
    """
    Configuration for UniMol encoder (molecule or pocket).

    From the UniMol paper: https://openreview.net/forum?id=6K2RM6wVqKu
    """

    model_type = "unimol"

    def __init__(
        self,
        vocab_size: int = 31,
        hidden_size: int = 512,
        num_hidden_layers: int = 15,
        num_attention_heads: int = 64,
        intermediate_size: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        activation_dropout: float = 0.0,
        max_position_embeddings: int = 512,
        gbf_kernels: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.gbf_kernels = gbf_kernels