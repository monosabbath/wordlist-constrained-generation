import sys
import transformers.tokenization_utils_base
import types

# Monkey patch for lm-format-enforcer compatibility with transformers v5
utils = types.ModuleType("transformers.tokenization_utils")
utils.PreTrainedTokenizerBase = transformers.tokenization_utils_base.PreTrainedTokenizerBase
sys.modules['transformers.tokenization_utils'] = utils
