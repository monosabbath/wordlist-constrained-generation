"""
Vocabulary constraints for controlled text generation.

This module provides functions to:
- Build regex-based prefix constraints from wordlists
- Get stop token IDs for generation termination
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import wordlist_generation  # noqa: F401  # Ensure monkey patch is applied
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from wordlist_generation.inference.vocab_constraints.trie import (
    TrieNode,
    build_trie_with_ranks,
    trie_to_regex,
    normalize_word,
)


# --- Stop Token Functions ---

def _safe_add(tok, token_name: str, ids: set):
    """Safely add a token ID to the set if it exists and is valid."""
    try:
        eid = tok.convert_tokens_to_ids(token_name)
        if eid is not None and eid != tok.unk_token_id and eid != -1:
            ids.add(int(eid))
    except Exception:
        pass


def get_stop_ids(tokenizer) -> List[int]:
    """Get all stop token IDs for the given tokenizer."""
    stop_ids: set[int] = set()

    # Add EOS token(s)
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, int):
            stop_ids.add(tokenizer.eos_token_id)
        elif isinstance(tokenizer.eos_token_id, list):
            for i in tokenizer.eos_token_id:
                stop_ids.add(int(i))

    # Add common end markers
    common_end_markers = (
        "<end_of_turn>", "<|eot_id|>", "<|im_end|>",
        "<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"
    )
    for special in common_end_markers:
        _safe_add(tokenizer, special, stop_ids)

    return list(stop_ids)



# --- Trie and Regex Caches ---

TRIECACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}  # key: (wordlist_dir, lang)
_PREFIX_FN_CACHE: Dict[Tuple[str, str, int, str], Any] = {}


# --- Helper Functions ---

def _safe_lang_name(lang: str) -> bool:
    """Validate language name contains only safe characters."""
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", lang))


def get_or_build_trie(lang: str, wordlist_dir: str) -> Optional[Dict[str, Any]]:
    """Get or build a trie for the given language wordlist."""
    if not _safe_lang_name(lang):
        return None
    key = (wordlist_dir, lang)
    if key in TRIECACHE:
        return TRIECACHE[key]
    filename = os.path.join(wordlist_dir, f"{lang}.txt")
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, encoding="utf-8") as fin:
            words = [normalize_word(w) for w in fin if w.strip()]
    except Exception:
        return None
    if not words:
        return None
    trie = build_trie_with_ranks(words)
    TRIECACHE[key] = {"trie": trie}
    return TRIECACHE[key]


def build_word_regex_for_n(lang: str, n_words: int, wordlist_dir: str) -> Optional[str]:
    """Build a regex pattern for the top n words in the language wordlist."""
    data = get_or_build_trie(lang, wordlist_dir)
    if data is None:
        return None
    trie: TrieNode = data["trie"]
    if trie.min_rank > n_words:
        return None
    return trie_to_regex(trie, nlimit=n_words)


# --- Main Prefix Function Builder ---

def _prefix_cache_key(tokenizer, lang: str, n_words: int, wordlist_dir: str) -> Tuple[str, str, int, str]:
    """Create a cache key for the prefix function."""
    name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    eos = str(getattr(tokenizer, "eos_token_id", "None"))
    return (name, eos, n_words, f"{wordlist_dir}:{lang}")


def build_regexp_prefix_fn(
    tokenizer,
    lang: str,
    n_words: int,
    wordlist_dir: str,
):
    """
    Build a prefix-constrained token function for vocabulary-limited generation.

    Args:
        tokenizer: The tokenizer to use
        lang: Language code (must match a wordlist file)
        n_words: Number of top words to allow from the wordlist
        wordlist_dir: Directory containing wordlist files

    Returns:
        A function that can be passed as `prefix_allowed_tokens_fn` to model.generate(),
        or None if the wordlist cannot be loaded.
    """
    key = _prefix_cache_key(tokenizer, lang, n_words, wordlist_dir)
    if key in _PREFIX_FN_CACHE:
        return _PREFIX_FN_CACHE[key]

    if get_or_build_trie(lang, wordlist_dir) is None:
        return None
    word_regex = build_word_regex_for_n(lang, n_words, wordlist_dir)
    if not word_regex:
        return None

    # Allow words separated by punctuation/whitespace; flexible boundaries
    punct_regex = r'[.,!?¿¡…\s]+'
    flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex}{punct_regex})*'

    parser = RegexParser(flexible_grammar)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    stop_ids = set(get_stop_ids(tokenizer))

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        # Only allow stopping when the regex is at an accepting state
        # (i.e. after a complete word + punctuation), ensuring all stop
        # token variants are available at those points.
        if stop_ids & allowed:
            allowed |= stop_ids
        return list(allowed)

    _PREFIX_FN_CACHE[key] = wrapped_prefix_fn
    return wrapped_prefix_fn
