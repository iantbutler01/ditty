from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _replicated_dtensor_from_local(local: torch.Tensor, like: Any) -> torch.Tensor:
    from torch.distributed.tensor import DTensor, Replicate

    dtensor = like if isinstance(like, DTensor) else getattr(like, "data", None)
    if not isinstance(dtensor, DTensor):
        return local
    placements = [Replicate()] * dtensor.device_mesh.ndim
    return DTensor.from_local(local, device_mesh=dtensor.device_mesh, placements=placements)


def _dtensor_to_local_replicated(value: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor, Replicate

    if not isinstance(value, DTensor):
        return value
    placements = [Replicate()] * value.device_mesh.ndim
    return value.redistribute(placements=placements).to_local()


def materialize_model_tensor(
    tensor: torch.Tensor | None,
    *,
    model: Any = None,
    attr_path: str | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None

    if (
        model is not None
        and attr_path
        and hasattr(tensor, "untyped_storage")
        and tensor.untyped_storage().size() == 0
    ):
        obj = getattr(model, "_orig_mod", model)
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        tensor = obj

    from torch.distributed.tensor import DTensor, Replicate

    if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim).to_local()
    elif hasattr(tensor, "data") and isinstance(tensor.data, DTensor):
        tensor = tensor.data.redistribute(
            placements=[Replicate()] * tensor.data.device_mesh.ndim
        ).to_local()
    return tensor


def extract_last_hidden_state(prediction: Any) -> torch.Tensor:
    if isinstance(prediction, torch.Tensor):
        return prediction
    if hasattr(prediction, "last_hidden_state") and isinstance(
        prediction.last_hidden_state, torch.Tensor
    ):
        return prediction.last_hidden_state
    hidden_states = getattr(prediction, "hidden_states", None)
    if hidden_states and isinstance(hidden_states[-1], torch.Tensor):
        return hidden_states[-1]
    if isinstance(prediction, dict):
        if isinstance(prediction.get("last_hidden_state"), torch.Tensor):
            return prediction["last_hidden_state"]
        hidden_states = prediction.get("hidden_states")
        if hidden_states and isinstance(hidden_states[-1], torch.Tensor):
            return hidden_states[-1]
    raise TypeError(
        "Could not extract hidden states from model output. "
        f"Expected a tensor or an object with `.last_hidden_state`, got {type(prediction).__name__}."
    )


def resolve_output_embeddings(model: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    real_model = getattr(model, "_orig_mod", model)
    if hasattr(real_model, "get_output_embeddings"):
        output_embeddings = real_model.get_output_embeddings()
    else:
        output_embeddings = getattr(real_model, "lm_head", None)
    if output_embeddings is None:
        raise RuntimeError(
            f"Model {type(real_model).__name__} does not expose output embeddings via "
            "`get_output_embeddings()` or `lm_head`."
        )

    weight = materialize_model_tensor(getattr(output_embeddings, "weight", None))
    bias = materialize_model_tensor(getattr(output_embeddings, "bias", None))
    if weight is None:
        raise RuntimeError(
            f"Output embeddings for model {type(real_model).__name__} did not expose a weight tensor."
        )
    return weight, bias


def resolve_output_projection(model: Any):
    real_model = getattr(model, "_orig_mod", model)
    if hasattr(real_model, "get_output_embeddings"):
        output_embeddings = real_model.get_output_embeddings()
    else:
        output_embeddings = getattr(real_model, "lm_head", None)
    if output_embeddings is None:
        raise RuntimeError(
            f"Model {type(real_model).__name__} does not expose output embeddings via "
            "`get_output_embeddings()` or `lm_head`."
        )
    return output_embeddings


def gather_selected_logprobs_from_hidden(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    output_projection=None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected hidden states with shape [batch, seq, hidden], got {tuple(hidden_states.shape)}"
        )
    if labels.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"Expected labels with shape {tuple(hidden_states.shape[:2])}, got {tuple(labels.shape)}"
        )

    labels_clamped = labels.clamp_min(0)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels_clamped.reshape(-1)
    flat_valid = valid_mask.reshape(-1).bool()

    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return torch.zeros_like(labels, dtype=hidden_states.dtype)

    flat_output = torch.zeros_like(flat_labels, dtype=hidden_states.dtype)
    effective_chunk_size = max(int(chunk_size), 1)

    for chunk_indices in valid_indices.split(effective_chunk_size):
        hidden_chunk = flat_hidden.index_select(0, chunk_indices)
        label_chunk = flat_labels.index_select(0, chunk_indices)
        if output_projection is not None:
            weight = getattr(output_projection, "weight", None)
            logits_chunk = output_projection(_replicated_dtensor_from_local(hidden_chunk, weight))
            logits_chunk = _dtensor_to_local_replicated(logits_chunk)
        else:
            if weight is None:
                raise ValueError("weight is required when output_projection is not provided")
            logits_chunk = F.linear(hidden_chunk, weight, bias)
        selected_chunk = logits_chunk.gather(dim=-1, index=label_chunk.unsqueeze(-1)).squeeze(-1)
        logprob_chunk = selected_chunk - torch.logsumexp(logits_chunk, dim=-1)
        flat_output.scatter_(0, chunk_indices, logprob_chunk.to(dtype=flat_output.dtype))

    return flat_output.view_as(labels)
