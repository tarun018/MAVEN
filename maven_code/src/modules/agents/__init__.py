REGISTRY = {}

from .rnn_agent import RNNAgent
from .noise_rnn_agent import RNNAgent as NoiseRNNAgent
from .ff_agent import FFAgent
from .noise_rnn_agent_deep import RNNAgentDeep as NoiseRNNAgentDeep

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["noise_rnn"] = NoiseRNNAgent
REGISTRY["noise_rnn_deep"] = NoiseRNNAgentDeep