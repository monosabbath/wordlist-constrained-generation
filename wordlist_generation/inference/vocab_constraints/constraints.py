"""
Vocabulary constraints for controlled text generation.

This module provides functions to:
- Build regex-based prefix constraints from wordlists
- Get stop token IDs for generation termination

Grammar is language-conditional:
- Spaceless (zh, ja): words and punctuation alternate freely, no separator
- Elision (fr, it): apostrophe-ending words attach directly to the next word
- Contraction (en): apostrophe-starting words attach directly to the preceding word
- Standard (all others): mandatory punctuation/whitespace separator between words
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


# --- Language-Conditional Grammar ---

# No spaces between words — words and punctuation combine freely
_SPACELESS_LANGS = frozenset({'zh', 'ja'})

# Apostrophe elision — words ending in ' attach directly to next word
_ELISION_LANGS = frozenset({'fr', 'it'})

# Suffix contraction — words starting with ' attach directly to preceding word
_CONTRACTION_LANGS = frozenset({'en'})

# Base punctuation included for all languages (literal inside [...])
_BASE_PUNCT = '.,!?…'

# Additional punctuation per language family (base_lang prefix)
_EXTRA_PUNCT: Dict[str, str] = {
    'es': '¿¡',
    'ar': '،؟',
    'hi': '।',
    'zh': '，。！？、',
    'ja': '，。、！？',
}


def _base_lang(lang: str) -> str:
    """Extract base language from locale code (e.g., 'es-ES' -> 'es')."""
    return lang.split('-')[0]


def _punct_regex(lang: str) -> str:
    """Build punctuation+whitespace character class for the given language."""
    extra = _EXTRA_PUNCT.get(_base_lang(lang), '')
    return f'[{_BASE_PUNCT}{extra}\\s]+'


def _build_elision_grammar(
    lang: str, n_words: int, wordlist_dir: str, punct: str,
) -> Optional[str]:
    """Build grammar for languages with apostrophe elision (French, Italian).

    Splits the wordlist into apostrophe-ending words (c', l', j', ...) and
    all others. Apostrophe words attach directly to the following word without
    a separator; non-apostrophe words require a separator.

    Grammar: (?:punct)?(?:(?:word_apos)*word_other punct)*
    """
    data = get_or_build_trie(lang, wordlist_dir)
    if data is None:
        return None
    words = data["words"]

    apos_words = [w for w in words[:n_words] if w.endswith("'")]
    other_words = [w for w in words[:n_words] if not w.endswith("'")]

    if not other_words:
        # Degenerate case — fall back to spaceless grammar
        word_regex = build_word_regex_for_n(lang, n_words, wordlist_dir)
        return f'(?:{punct}|{word_regex})*'

    other_trie = build_trie_with_ranks(other_words)
    other_regex = trie_to_regex(other_trie, nlimit=len(other_words))

    if not apos_words:
        # No elision words in top n_words — standard grammar
        return f'(?:{punct})?(?:{other_regex}{punct})*'

    apos_trie = build_trie_with_ranks(apos_words)
    apos_regex = trie_to_regex(apos_trie, nlimit=len(apos_words))

    return f'(?:{punct})?(?:(?:{apos_regex})*{other_regex}{punct})*'


def _build_contraction_grammar(
    lang: str, n_words: int, wordlist_dir: str, punct: str,
    word_regex: str,
) -> Optional[str]:
    """Build grammar for languages with suffix contractions (English).

    Splits the wordlist into apostrophe-starting words ('s, 't, 're, ...) and
    all others. Apostrophe words attach directly to the preceding word without
    a separator; non-apostrophe words require a separator.

    The full word regex (including apostrophe words) is used as the main word
    pattern so apostrophe words can also appear standalone after a separator.

    Grammar: (?:punct)?(?:word_all(?:word_apos)*punct)*
    """
    data = get_or_build_trie(lang, wordlist_dir)
    if data is None:
        return None
    words = data["words"]

    apos_words = [w for w in words[:n_words] if w.startswith("'")]

    if not apos_words:
        # No contraction words in top n_words — standard grammar
        return f'(?:{punct})?(?:{word_regex}{punct})*'

    apos_trie = build_trie_with_ranks(apos_words)
    apos_regex = trie_to_regex(apos_trie, nlimit=len(apos_words))

    return f'(?:{punct})?(?:{word_regex}(?:{apos_regex})*{punct})*'


def _build_grammar(
    word_regex: str, lang: str, n_words: int, wordlist_dir: str,
) -> Optional[str]:
    """Build the constraint grammar appropriate for the language."""
    punct = _punct_regex(lang)
    base = _base_lang(lang)

    if base in _SPACELESS_LANGS:
        return f'(?:{punct}|{word_regex})*'

    if base in _ELISION_LANGS:
        return _build_elision_grammar(lang, n_words, wordlist_dir, punct)

    if base in _CONTRACTION_LANGS:
        return _build_contraction_grammar(lang, n_words, wordlist_dir, punct, word_regex)

    # Standard: mandatory separator between words
    return f'(?:{punct})?(?:{word_regex}{punct})*'


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
        "<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>",
        "<turn|>",  # Gemma 4 end-of-turn
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
    TRIECACHE[key] = {"trie": trie, "words": words}
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

    flexible_grammar = _build_grammar(word_regex, lang, n_words, wordlist_dir)
    if flexible_grammar is None:
        return None

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
