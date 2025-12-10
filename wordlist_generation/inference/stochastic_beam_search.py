"""
Stochastic Beam Search with Probabilistic Beam Pruning

This module provides a custom beam search implementation that adds stochastic
beam pruning. When do_sample=True, instead of greedily selecting the top num_beams
beams, it samples which beams to keep based on their scores.

Usage:
    from wordlist_generation.inference.stochastic_beam_search import stochastic_beam_search_generate

    outputs = model.generate(
        **inputs,
        custom_generate=stochastic_beam_search_generate,
        beam_pruning_temperature=1.0,  # Controls randomness of beam selection
        **other_gen_kwargs
    )

Based on transformers v4.57.x beam search implementation.
"""

from typing import Optional, Union

import torch
from torch import nn

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import ModelOutput


# Output classes for beam search
class GenerateBeamOutput(ModelOutput):
    """Base class for beam search outputs."""
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[tuple] = None
    logits: Optional[tuple] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    past_key_values: Optional[tuple] = None


class GenerateBeamDecoderOnlyOutput(GenerateBeamOutput):
    """Output for decoder-only beam search."""
    pass


class GenerateBeamEncoderDecoderOutput(GenerateBeamOutput):
    """Output for encoder-decoder beam search."""
    encoder_attentions: Optional[tuple] = None
    encoder_hidden_states: Optional[tuple] = None
    decoder_attentions: Optional[tuple] = None
    cross_attentions: Optional[tuple] = None
    decoder_hidden_states: Optional[tuple] = None


# Helper functions (copied from GenerationMixin)
def _flatten_beam_dim(tensor: torch.Tensor) -> torch.Tensor:
    """[batch_size, num_beams, ...] -> [batch_size * num_beams, ...]"""
    shape = list(tensor.shape)
    return torch.reshape(tensor, [shape[0] * shape[1]] + shape[2:])


def _unflatten_beam_dim(tensor: torch.Tensor, batch_size: int, num_beams: int) -> torch.Tensor:
    """[batch_size * num_beams, ...] -> [batch_size, num_beams, ...]"""
    shape = list(tensor.shape)
    return torch.reshape(tensor, [batch_size, num_beams] + shape[1:])


def _gather_beams(tensor: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
    """
    Gathers the beam slices indexed by beam_indices into new beam array.
    """
    while len(beam_indices.shape) < len(tensor.shape):
        beam_indices = beam_indices.unsqueeze(-1)
    gathered_tensor = torch.take_along_dim(input=tensor, indices=beam_indices, dim=1)
    return gathered_tensor


def _check_early_stop_heuristic(
    is_early_stop_heuristic_unsatisfied: torch.Tensor,
    running_beam_scores: torch.Tensor,
    beam_scores: torch.Tensor,
    is_sent_finished: torch.Tensor,
    cur_len: int,
    max_length: int,
    decoder_prompt_len: int,
    early_stopping: Union[bool, str],
    length_penalty: float,
):
    """
    Determine whether early stopping is possible by checking if the best possible score
    of running beams could still improve upon the finished ones.
    """
    if early_stopping == "never" and length_penalty > 0.0:
        best_hypothetical_length = max_length - decoder_prompt_len
    else:
        best_hypothetical_length = cur_len - decoder_prompt_len
    best_possible_running_score = running_beam_scores[:, :1] / (best_hypothetical_length**length_penalty)
    worst_finished_score = torch.where(is_sent_finished, torch.min(beam_scores, dim=1, keepdim=True)[0], -1.0e9)
    return is_early_stop_heuristic_unsatisfied & torch.any(
        best_possible_running_score > worst_finished_score, dim=-1, keepdim=True
    )


def _beam_search_has_unfinished_sequences(
    is_early_stop_heuristic_unsatisfied: torch.Tensor,
    is_sent_finished: torch.Tensor,
    next_token_hits_stopping_criteria: torch.Tensor,
    early_stopping: Union[bool, str],
):
    """
    Beam Search stopping condition -- halts the generation loop if any of these conditions becomes False
    """
    improvement_possible = torch.any(is_early_stop_heuristic_unsatisfied)
    exists_open_beam = ~(torch.all(is_sent_finished) & (early_stopping is True))
    valid_continuations = ~torch.all(next_token_hits_stopping_criteria)
    return improvement_possible & exists_open_beam & valid_continuations


def _get_top_k_continuations(
    accumulated_log_probs: torch.Tensor,
    running_sequences: torch.Tensor,
    running_beam_indices: torch.Tensor,
    cur_len: int,
    decoder_prompt_len: int,
    do_sample: bool,
    beams_to_keep: int,
    num_beams: int,
    vocab_size: int,
    batch_size: int,
) -> tuple:
    """
    Get top-K continuations given the accumulated log probs on the next token.
    """
    # Gather the top K scores from _all_ beams.
    if do_sample:
        topk_indices = torch.multinomial(
            nn.functional.softmax(accumulated_log_probs, dim=-1), num_samples=beams_to_keep
        )
        topk_log_probs = torch.gather(input=accumulated_log_probs, dim=1, index=topk_indices)
    else:
        topk_log_probs, topk_indices = torch.topk(accumulated_log_probs, k=beams_to_keep)

    # Gather K top beams, recover the beam index by floor division and token id by modulo division
    topk_current_beam_indices = topk_indices // vocab_size
    topk_running_beam_indices = _gather_beams(running_beam_indices, topk_current_beam_indices)
    topk_running_sequences = _gather_beams(running_sequences, topk_current_beam_indices)
    topk_ids = topk_indices % vocab_size

    # Update sequences for the K top-k new sequences.
    topk_running_sequences[:, :, cur_len] = topk_ids

    # we want to store the beam indices with batch information
    batch_offset = torch.arange(batch_size, device=topk_ids.device).view(-1, 1) * num_beams
    batch_modified_indices = topk_current_beam_indices + batch_offset
    topk_running_beam_indices[:, :, cur_len - decoder_prompt_len] = batch_modified_indices

    return topk_log_probs, topk_running_sequences, topk_running_beam_indices


def _get_running_beams_for_next_iteration_stochastic(
    topk_log_probs: torch.Tensor,
    topk_running_sequences: torch.Tensor,
    topk_running_beam_indices: torch.Tensor,
    next_token_hits_stopping_criteria: torch.Tensor,
    num_beams: int,
    do_sample: bool,
    beam_pruning_temperature: float = 1.0,
) -> tuple:
    """
    Given the top-K continuations, their scores, and whether they hit a stopping criteria,
    select beams for the next iteration.

    MODIFICATION: When do_sample=True, uses stochastic sampling instead of greedy top-k.
    """
    # To prevent finished sequences from being used in subsequent iterations,
    # set their log probs to a very large negative value
    topk_running_log_probs = topk_log_probs + next_token_hits_stopping_criteria.to(torch.float32) * -1.0e9

    if do_sample and beam_pruning_temperature > 0:
        # STOCHASTIC BEAM SELECTION: Sample beams based on their scores
        # Apply temperature scaling and convert to probabilities
        scaled_log_probs = topk_running_log_probs / beam_pruning_temperature
        probs = nn.functional.softmax(scaled_log_probs, dim=-1)
        next_topk_indices = torch.multinomial(probs, num_samples=num_beams, replacement=False)
    else:
        # Greedy selection (standard behavior, or when temperature <= 0)
        next_topk_indices = torch.topk(topk_running_log_probs, k=num_beams)[1]

    running_sequences = _gather_beams(topk_running_sequences, next_topk_indices)
    running_beam_scores = _gather_beams(topk_running_log_probs, next_topk_indices)
    running_beam_indices = _gather_beams(topk_running_beam_indices, next_topk_indices)
    return running_sequences, running_beam_scores, running_beam_indices


def _update_finished_beams(
    sequences: torch.Tensor,
    topk_running_sequences: torch.Tensor,
    beam_scores: torch.Tensor,
    topk_log_probs: torch.Tensor,
    beam_indices: torch.Tensor,
    topk_running_beam_indices: torch.Tensor,
    is_early_stop_heuristic_unsatisfied: torch.Tensor,
    is_sent_finished: torch.Tensor,
    next_token_hits_stopping_criteria: torch.Tensor,
    top_num_beam_mask: torch.Tensor,
    num_beams: int,
    cur_len: int,
    decoder_prompt_len: int,
    length_penalty: float,
    early_stopping: Union[bool, str],
) -> tuple:
    """
    Updates the finished beams if there are new completed sequences with higher scores.
    """
    did_top_num_beams_just_finished = next_token_hits_stopping_criteria & top_num_beam_mask[None, :]

    # Apply length penalty
    topk_log_probs = topk_log_probs / ((cur_len + 1 - decoder_prompt_len) ** length_penalty)
    # Make sure no scores can be added if beam is full and early stopping is on
    beams_in_batch_are_full = torch.all(is_sent_finished, axis=-1, keepdims=True) & (early_stopping is True)
    topk_log_probs = topk_log_probs + beams_in_batch_are_full.to(torch.float32) * -1.0e9
    # Make sure no scores can be added if improvement is not possible
    topk_log_probs = topk_log_probs + (~is_early_stop_heuristic_unsatisfied).to(torch.float32) * -1.0e9
    # Make sure still running sequences cannot be chosen as finalized
    topk_log_probs = topk_log_probs + (~did_top_num_beams_just_finished) * -1.0e9

    # Merge and keep best
    merged_sequences = torch.cat((sequences, topk_running_sequences), dim=1)
    merged_scores = torch.cat((beam_scores, topk_log_probs), dim=1)
    merged_beam_indices = torch.cat((beam_indices, topk_running_beam_indices), dim=1)
    merged_is_sent_finished = torch.cat((is_sent_finished, did_top_num_beams_just_finished), dim=1)
    topk_merged_indices = torch.topk(merged_scores, k=num_beams)[1]
    sequences = _gather_beams(merged_sequences, topk_merged_indices)
    beam_scores = _gather_beams(merged_scores, topk_merged_indices)
    beam_indices = _gather_beams(merged_beam_indices, topk_merged_indices)
    is_sent_finished = _gather_beams(merged_is_sent_finished, topk_merged_indices)
    return sequences, beam_scores, beam_indices, is_sent_finished


def stochastic_beam_search_generate(
    model,
    input_ids: torch.LongTensor,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    beam_pruning_temperature: float = 1.0,
    synced_gpus: bool = False,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    """
    Custom beam search with stochastic beam pruning.

    When do_sample=True, instead of greedily keeping the top num_beams beams,
    this implementation samples which beams to keep based on their scores.

    Args:
        model: The model to use for generation.
        input_ids: The input token IDs.
        generation_config: Generation configuration.
        logits_processor: List of logits processors.
        stopping_criteria: List of stopping criteria.
        beam_pruning_temperature: Temperature for stochastic beam selection.
            Higher values = more random, lower values = closer to greedy.
            Default: 1.0
        synced_gpus: Whether to sync GPUs (for distributed training).
        **model_kwargs: Additional model keyword arguments.

    Returns:
        Generated sequences (and optionally scores/attentions if configured).
    """
    # Initialize defaults
    if generation_config is None:
        generation_config = model.generation_config
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList()

    # Extract generation parameters
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    do_sample = generation_config.do_sample
    early_stopping = generation_config.early_stopping
    length_penalty = generation_config.length_penalty
    max_length = generation_config.max_length
    num_beams = generation_config.num_beams
    num_return_sequences = generation_config.num_return_sequences

    batch_size_unflattened, cur_len = input_ids.shape[:2]
    batch_size = batch_size_unflattened // num_beams

    # Get vocab size
    if model.__class__.__name__ == "MoshiDepthDecoder":
        vocab_size = model.config.audio_vocab_size
    elif model.__class__.__name__ == "ImageGPTForCausalImageModeling":
        vocab_size = model.get_output_embeddings().out_features
    elif model.__class__.__name__ == "BarkSemanticModel":
        vocab_size = model.config.output_vocab_size
    else:
        vocab_size = model.config.get_text_config().vocab_size

    decoder_prompt_len = cur_len
    this_peer_finished = False

    # Calculate beams_to_keep
    n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
    beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
    top_num_beam_mask = torch.cat(
        (torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
        dim=0,
    ).to(input_ids.device)

    # Initialize cache position
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    # Check for unsupported low_memory mode
    if generation_config.low_memory:
        raise ValueError("`low_memory=True` is not supported in stochastic beam search.")

    # Initialize output lists (use lists to avoid tuple concatenation overhead)
    all_scores = [] if (return_dict_in_generate and output_scores) else None
    raw_logits = [] if (return_dict_in_generate and output_logits) else None
    decoder_attentions = [] if (return_dict_in_generate and output_attentions) else None
    cross_attentions = [] if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = [] if (return_dict_in_generate and output_hidden_states) else None

    # Encoder-decoder specific
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # Initialize running tensors
    output_fill_value = pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
    running_sequences = torch.full(
        (batch_size, num_beams, max_length),
        fill_value=output_fill_value,
        dtype=torch.int64,
        device=input_ids.device,
    )
    running_sequences[:, :, :cur_len] = _unflatten_beam_dim(input_ids, batch_size, num_beams)
    sequences = running_sequences.clone()

    # Initialize scores - first beam gets 0, rest get -1e9
    running_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    running_beam_scores[:, 1:] = -1e9
    beam_scores = torch.full((batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=input_ids.device)

    # State tracking
    is_sent_finished = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=input_ids.device)
    is_early_stop_heuristic_unsatisfied = torch.ones((batch_size, 1), dtype=torch.bool, device=input_ids.device)
    next_token_hits_stopping_criteria = torch.zeros(
        (batch_size, num_beams), dtype=torch.bool, device=input_ids.device
    )

    # Beam indices tracking
    running_beam_indices = torch.full(
        (batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=torch.int32, device=input_ids.device
    )
    beam_indices = running_beam_indices.clone()

    # Generation loop
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # Forward pass
        flat_running_sequences = _flatten_beam_dim(running_sequences[:, :, :cur_len])
        model_inputs = model.prepare_inputs_for_generation(flat_running_sequences, **model_kwargs)
        model_outputs = model(**model_inputs, return_dict=True)

        # Update model kwargs
        model_kwargs = model._update_model_kwargs_for_generation(
            model_outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Get logits and compute log probs
        logits = model_outputs.logits[:, -1, :].to(dtype=torch.float32, device=input_ids.device)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs = logits_processor(flat_running_sequences, log_probs)

        # Store outputs if requested
        if return_dict_in_generate:
            if output_logits:
                raw_logits.append(logits.clone())
            if output_scores:
                all_scores.append(log_probs.clone())
            if output_attentions:
                decoder_attentions.append(
                    model_outputs.decoder_attentions
                    if model.config.is_encoder_decoder
                    else model_outputs.attentions
                )
                if model.config.is_encoder_decoder:
                    cross_attentions.append(model_outputs.cross_attentions)
            if output_hidden_states:
                decoder_hidden_states.append(
                    model_outputs.decoder_hidden_states
                    if model.config.is_encoder_decoder
                    else model_outputs.hidden_states
                )

        del model_outputs

        # Accumulate scores
        log_probs = _unflatten_beam_dim(log_probs, batch_size, num_beams)
        log_probs = log_probs + running_beam_scores[:, :, None]
        log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

        # Get top-K continuations
        topk_log_probs, topk_running_sequences, topk_running_beam_indices = _get_top_k_continuations(
            accumulated_log_probs=log_probs,
            running_sequences=running_sequences,
            running_beam_indices=running_beam_indices,
            cur_len=cur_len,
            decoder_prompt_len=decoder_prompt_len,
            do_sample=do_sample,
            beams_to_keep=beams_to_keep,
            num_beams=num_beams,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )

        # Check stopping criteria
        next_token_hits_stopping_criteria = stopping_criteria(
            _flatten_beam_dim(topk_running_sequences[:, :, : cur_len + 1]),
            all_scores,
        )
        next_token_hits_stopping_criteria = _unflatten_beam_dim(
            next_token_hits_stopping_criteria, batch_size, beams_to_keep
        )

        # Get running beams for next iteration (STOCHASTIC VERSION)
        running_sequences, running_beam_scores, running_beam_indices = _get_running_beams_for_next_iteration_stochastic(
            topk_log_probs=topk_log_probs,
            topk_running_sequences=topk_running_sequences,
            topk_running_beam_indices=topk_running_beam_indices,
            next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            beam_pruning_temperature=beam_pruning_temperature,
        )

        # Update finished beams
        sequences, beam_scores, beam_indices, is_sent_finished = _update_finished_beams(
            sequences=sequences,
            topk_running_sequences=topk_running_sequences,
            beam_scores=beam_scores,
            topk_log_probs=topk_log_probs,
            beam_indices=beam_indices,
            topk_running_beam_indices=topk_running_beam_indices,
            is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
            is_sent_finished=is_sent_finished,
            next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
            top_num_beam_mask=top_num_beam_mask,
            num_beams=num_beams,
            cur_len=cur_len,
            decoder_prompt_len=decoder_prompt_len,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )

        # Reorder cache
        if model_kwargs.get("past_key_values", None) is not None:
            beam_idx = _flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len])
            if hasattr(model, "_reorder_cache"):
                model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)
            else:
                model_kwargs["past_key_values"].reorder_cache(beam_idx)

        cur_len = cur_len + 1
        is_early_stop_heuristic_unsatisfied = _check_early_stop_heuristic(
            is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
            running_beam_scores=running_beam_scores,
            beam_scores=beam_scores,
            is_sent_finished=is_sent_finished,
            cur_len=cur_len,
            max_length=max_length,
            decoder_prompt_len=decoder_prompt_len,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )
        this_peer_finished = not _beam_search_has_unfinished_sequences(
            is_early_stop_heuristic_unsatisfied,
            is_sent_finished,
            next_token_hits_stopping_criteria,
            early_stopping,
        )

    # Prepare outputs - stochastic final selection if do_sample and temperature > 0
    if do_sample and beam_pruning_temperature > 0 and num_return_sequences <= num_beams:
        # Sample which sequences to return based on their scores
        # beam_scores shape: [batch_size, num_beams]
        selection_probs = nn.functional.softmax(beam_scores / beam_pruning_temperature, dim=-1)

        # Sample num_return_sequences indices without replacement
        selected_indices = torch.multinomial(selection_probs, num_samples=num_return_sequences, replacement=False)

        # Gather selected sequences
        sequences = _gather_beams(sequences, selected_indices)
        beam_scores = _gather_beams(beam_scores, selected_indices)
        beam_indices = _gather_beams(beam_indices, selected_indices)

    sequences = _flatten_beam_dim(sequences[:, :num_return_sequences, :])
    beam_scores = _flatten_beam_dim(beam_scores[:, :num_return_sequences])
    beam_indices = _flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

    # Crop to actual size
    max_generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
    output_length = decoder_prompt_len + max_generated_length
    sequences = sequences[:, :output_length]
    beam_indices = beam_indices[:, :max_generated_length]

    if return_dict_in_generate:
        if not output_scores:
            beam_scores = None

        # Convert lists to tuples for output
        final_scores = tuple(all_scores) if all_scores else None
        final_logits = tuple(raw_logits) if raw_logits else None
        final_decoder_attentions = tuple(decoder_attentions) if decoder_attentions else None
        final_cross_attentions = tuple(cross_attentions) if cross_attentions else None
        final_decoder_hidden_states = tuple(decoder_hidden_states) if decoder_hidden_states else None

        if model.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequences,
                sequences_scores=beam_scores,
                scores=final_scores,
                logits=final_logits,
                beam_indices=beam_indices,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=final_decoder_attentions,
                cross_attentions=final_cross_attentions,
                decoder_hidden_states=final_decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequences,
                sequences_scores=beam_scores,
                scores=final_scores,
                logits=final_logits,
                beam_indices=beam_indices,
                attentions=final_decoder_attentions,
                hidden_states=final_decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequences
