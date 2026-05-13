from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from modeling.reason import LatentTransformerReasoningModel, TransformerReasoningNet


@dataclass
class AdaptiveReasoningOutput:
    latent_trajectory: torch.Tensor
    raw_latent_trajectory: torch.Tensor
    latent_attention_mask: torch.Tensor
    chunk_mask: torch.Tensor
    active_chunks: torch.Tensor
    expected_active_chunks: torch.Tensor
    active_tokens: torch.Tensor
    expected_active_tokens: torch.Tensor
    budget_values: torch.Tensor


def _last_valid_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states[:, -1, :]
    mask = attention_mask.long()
    has_token = mask.any(dim=1)
    flipped = torch.flip(mask, dims=[1])
    distance_from_end = flipped.float().argmax(dim=1)
    last_indices = mask.size(1) - 1 - distance_from_end
    last_indices = torch.where(
        has_token,
        last_indices,
        torch.full_like(last_indices, mask.size(1) - 1),
    )
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices]


class AdaptiveTrajectoryAnchoredReasoningNet(TransformerReasoningNet):
    """Transformer reasoning net with a scalar prefix-budget router."""

    is_adaptive_reasoning_net = True

    def __init__(
        self,
        model_name_or_path,
        latent_trajectory_length: int = 384,
        hidden_size: int = 1024,
        chunk_size: int = 16,
        min_chunks: int = 2,
        router_tau: float = 1.0,
        init_chunks: int = 16,
    ):
        if latent_trajectory_length % chunk_size != 0:
            raise ValueError(
                "latent_trajectory_length must be divisible by chunk_size; "
                f"got {latent_trajectory_length=} and {chunk_size=}."
            )
        super().__init__(
            model_name_or_path,
            latent_trajectory_length=latent_trajectory_length,
            hidden_size=hidden_size,
        )
        self.chunk_size = chunk_size
        self.max_chunks = latent_trajectory_length // chunk_size
        self.min_chunks = min_chunks
        self.router_tau = router_tau

        if self.min_chunks < 1 or self.min_chunks > self.max_chunks:
            raise ValueError(
                f"min_chunks must be in [1, {self.max_chunks}], got {self.min_chunks}."
            )

        router_hidden = max(64, hidden_size // 4)
        self.router = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, router_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(router_hidden, 1),
        )
        self.router.to(self.reasoning_network.dtype)
        self._init_router_bias(init_chunks)

    def _init_router_bias(self, init_chunks: int) -> None:
        init_chunks = min(max(init_chunks, self.min_chunks), self.max_chunks)
        if self.max_chunks == self.min_chunks:
            bias = 0.0
        else:
            ratio = (init_chunks - self.min_chunks) / (self.max_chunks - self.min_chunks)
            ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
            bias = torch.logit(torch.tensor(ratio)).item()
        final_linear = self.router[-1]
        torch.nn.init.zeros_(final_linear.weight)
        torch.nn.init.constant_(final_linear.bias, bias)

    def _route_budget(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pooled = _last_valid_token_pool(hidden_states, attention_mask)
        pooled = pooled.to(self.reasoning_network.dtype)
        raw_budget = self.router(pooled).squeeze(-1)
        return self.min_chunks + torch.sigmoid(raw_budget) * (self.max_chunks - self.min_chunks)

    def _build_chunk_mask(
        self,
        budget_values: torch.Tensor,
        *,
        hard: bool,
        fixed_latent_chunks: Optional[int],
        tau: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fixed_latent_chunks is not None:
            active_chunks = torch.full(
                (budget_values.size(0),),
                int(fixed_latent_chunks),
                device=budget_values.device,
                dtype=torch.long,
            ).clamp(self.min_chunks, self.max_chunks)
            chunk_positions = torch.arange(
                self.max_chunks,
                device=budget_values.device,
                dtype=torch.long,
            )
            chunk_mask = (chunk_positions.unsqueeze(0) < active_chunks.unsqueeze(1)).to(
                budget_values.dtype
            )
            return chunk_mask, active_chunks

        chunk_thresholds = torch.arange(
            self.max_chunks,
            device=budget_values.device,
            dtype=budget_values.dtype,
        ) + 0.5

        if hard:
            active_chunks = torch.round(budget_values).long().clamp(
                self.min_chunks,
                self.max_chunks,
            )
            chunk_mask = (chunk_thresholds.unsqueeze(0) < active_chunks.unsqueeze(1)).to(
                budget_values.dtype
            )
            return chunk_mask, active_chunks

        tau = tau if tau is not None else self.router_tau
        chunk_mask = torch.sigmoid((budget_values.unsqueeze(1) - chunk_thresholds.unsqueeze(0)) / tau)
        active_chunks = torch.round(budget_values).long().clamp(
            self.min_chunks,
            self.max_chunks,
        )
        return chunk_mask, active_chunks

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_metadata: bool = False,
        hard: bool = False,
        fixed_latent_chunks: Optional[int] = None,
        tau: Optional[float] = None,
        trim_to_active: bool = False,
    ):
        raw_latent = super().forward(hidden_states, attention_mask=attention_mask)
        budget_values = self._route_budget(hidden_states, attention_mask)
        chunk_mask, active_chunks = self._build_chunk_mask(
            budget_values,
            hard=hard,
            fixed_latent_chunks=fixed_latent_chunks,
            tau=tau,
        )
        token_mask = chunk_mask.repeat_interleave(self.chunk_size, dim=1)
        latent = raw_latent * token_mask.unsqueeze(-1).to(raw_latent.dtype)

        latent_attention_mask = torch.ones(
            latent.size(0),
            latent.size(1),
            device=latent.device,
            dtype=torch.long,
        )

        if hard:
            latent_attention_mask = token_mask.to(torch.long)

        if hard and trim_to_active and latent.size(0) == 1:
            active_tokens = int(active_chunks.max().item()) * self.chunk_size
            latent = latent[:, :active_tokens, :]
            raw_latent = raw_latent[:, :active_tokens, :]
            latent_attention_mask = latent_attention_mask[:, :active_tokens]
            token_mask = token_mask[:, :active_tokens]
            chunk_mask = chunk_mask[:, : int(active_chunks.max().item())]

        output = AdaptiveReasoningOutput(
            latent_trajectory=latent,
            raw_latent_trajectory=raw_latent,
            latent_attention_mask=latent_attention_mask,
            chunk_mask=chunk_mask,
            active_chunks=active_chunks,
            expected_active_chunks=chunk_mask.sum(dim=1),
            active_tokens=active_chunks * self.chunk_size,
            expected_active_tokens=chunk_mask.sum(dim=1) * self.chunk_size,
            budget_values=budget_values,
        )
        if return_metadata:
            return output
        return output.latent_trajectory


def _resample_teacher_steps(
    step_embeddings: torch.Tensor,
    step_mask: torch.Tensor,
    target_steps: int,
) -> torch.Tensor:
    """Linearly resample variable teacher steps to target chunk count."""
    batch_size, max_steps, hidden_size = step_embeddings.shape
    anchors = step_embeddings.new_zeros(batch_size, target_steps, hidden_size)
    for batch_idx in range(batch_size):
        valid_count = int(step_mask[batch_idx].sum().item())
        if valid_count <= 0:
            continue
        valid_steps = step_embeddings[batch_idx, :valid_count].transpose(0, 1).unsqueeze(0)
        if valid_count == 1:
            anchors[batch_idx] = valid_steps.squeeze(0).transpose(0, 1).expand(target_steps, -1)
            continue
        resized = F.interpolate(
            valid_steps,
            size=target_steps,
            mode="linear",
            align_corners=True,
        )
        anchors[batch_idx] = resized.squeeze(0).transpose(0, 1)
    return anchors


class AdaptiveLatentTransformerReasoningModel(LatentTransformerReasoningModel):
    """Latent reasoning model with answer, trajectory-anchor, and budget losses."""

    def __init__(
        self,
        slow_reasoning_model,
        processor,
        reasoning_network: AdaptiveTrajectoryAnchoredReasoningNet,
        *,
        anchor_loss_weight: float = 0.0,
        budget_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            slow_reasoning_model=slow_reasoning_model,
            processor=processor,
            reasoning_network=reasoning_network,
            **kwargs,
        )
        self.anchor_loss_weight = anchor_loss_weight
        self.budget_loss_weight = budget_loss_weight
        self.budget_loss_scale = 1.0
        self.last_loss_metrics: dict[str, float] = {}

    def set_budget_loss_scale(self, scale: float) -> None:
        self.budget_loss_scale = float(scale)

    def _encode_teacher_steps(
        self,
        teacher_input_ids: torch.LongTensor,
        teacher_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_steps, max_len = teacher_input_ids.shape
        flat_ids = teacher_input_ids.reshape(batch_size * max_steps, max_len)
        flat_mask = teacher_attention_mask.reshape(batch_size * max_steps, max_len)
        empty_rows = flat_mask.sum(dim=1) == 0
        if empty_rows.any():
            flat_mask = flat_mask.clone()
            flat_ids = flat_ids.clone()
            flat_mask[empty_rows, 0] = 1
            flat_ids[empty_rows, 0] = self.pad_token_id
        flat_embeds = self.get_input_embeddings(flat_ids).to(self.slow_reasoning_model.dtype)
        outputs = self.slow_reasoning_model(
            inputs_embeds=flat_embeds,
            attention_mask=flat_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1].detach()
        mask = flat_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.reshape(batch_size, max_steps, -1)

    def _compute_anchor_loss(
        self,
        raw_latent_trajectory: torch.Tensor,
        chunk_mask: torch.Tensor,
        teacher_input_ids: Optional[torch.LongTensor],
        teacher_attention_mask: Optional[torch.Tensor],
        teacher_step_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            teacher_input_ids is None
            or teacher_attention_mask is None
            or teacher_step_mask is None
            or teacher_step_mask.sum().item() == 0
        ):
            return raw_latent_trajectory.new_zeros(())

        with torch.no_grad():
            step_embeddings = self._encode_teacher_steps(
                teacher_input_ids=teacher_input_ids,
                teacher_attention_mask=teacher_attention_mask,
            )
            anchors = _resample_teacher_steps(
                step_embeddings,
                teacher_step_mask,
                target_steps=self.reasoning_network.max_chunks,
            )

        batch_size = raw_latent_trajectory.size(0)
        hidden_size = raw_latent_trajectory.size(-1)
        latent_chunks = raw_latent_trajectory.reshape(
            batch_size,
            self.reasoning_network.max_chunks,
            self.reasoning_network.chunk_size,
            hidden_size,
        ).mean(dim=2)
        cosine_loss = 1.0 - F.cosine_similarity(latent_chunks.float(), anchors.float(), dim=-1)
        sample_has_teacher = (teacher_step_mask.sum(dim=1) > 0).to(chunk_mask.dtype)
        weights = chunk_mask.to(cosine_loss.dtype) * sample_has_teacher.unsqueeze(1)
        return (cosine_loss * weights).sum() / weights.sum().clamp(min=1.0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        teacher_input_ids: Optional[torch.LongTensor] = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        teacher_step_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1) + labels.size(1)),
                device=input_ids.device,
                dtype=torch.long,
            )

        with torch.no_grad():
            prompt_mask = attention_mask[:, :input_ids.size(1)]
            prompt_embeddings, prompt_hidden_states = self._prefill_prompt(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                position_ids=position_ids,
                **kwargs,
            )

        reasoning_output = self.reasoning_network(
            prompt_hidden_states,
            attention_mask=prompt_mask,
            return_metadata=True,
            hard=False,
        )
        latent_trajectory = reasoning_output.latent_trajectory
        latent_trajectory_mask = torch.ones(
            latent_trajectory.size(0),
            latent_trajectory.size(1),
            device=input_ids.device,
            dtype=torch.long,
        )

        label_embeddings = self.get_input_embeddings(labels).to(self.slow_reasoning_model.dtype)
        labels_mask = attention_mask[:, input_ids.size(1):]

        input_embeddings = torch.cat([prompt_embeddings, latent_trajectory, label_embeddings], dim=1)
        input_mask = torch.cat([prompt_mask, latent_trajectory_mask, labels_mask], dim=1).long()

        output_labels = labels.masked_fill(labels == self.pad_token_id, -100).long()
        output_labels = torch.cat(
            (
                prompt_embeddings.new_ones(output_labels.size(0), prompt_embeddings.size(1)).long() * -100,
                latent_trajectory.new_ones(output_labels.size(0), latent_trajectory.size(1)).long() * -100,
                output_labels,
            ),
            dim=1,
        ).long()

        outputs = self.slow_reasoning_model(
            inputs_embeds=input_embeddings,
            attention_mask=input_mask,
            labels=output_labels,
            return_dict=True,
            **kwargs,
        )

        answer_loss = outputs.loss
        anchor_loss = self._compute_anchor_loss(
            reasoning_output.raw_latent_trajectory,
            reasoning_output.chunk_mask,
            teacher_input_ids,
            teacher_attention_mask,
            teacher_step_mask,
        )
        budget_loss = (
            reasoning_output.expected_active_tokens / self.reasoning_network.latent_trajectory_length
        ).mean()
        total_loss = (
            answer_loss
            + self.anchor_loss_weight * anchor_loss
            + self.budget_loss_weight * self.budget_loss_scale * budget_loss
        )

        self.last_loss_metrics = {
            "answer_loss": float(answer_loss.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "budget_loss": float(budget_loss.detach().cpu()),
            "active_latent_tokens": float(reasoning_output.expected_active_tokens.detach().mean().cpu()),
        }

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        generation_config=None,
        fixed_latent_chunks: Optional[int] = None,
        return_latent_budget: bool = False,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1)),
                device=input_ids.device,
                dtype=torch.long,
            )

        prompt_mask = attention_mask.long()
        generation_kwargs = dict(kwargs)
        generation_kwargs.setdefault("use_cache", True)

        with torch.no_grad():
            prompt_embeddings, prompt_hidden_states = self._prefill_prompt(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                position_ids=position_ids,
            )
            reasoning_output = self.reasoning_network(
                prompt_hidden_states,
                attention_mask=prompt_mask,
                return_metadata=True,
                hard=True,
                fixed_latent_chunks=fixed_latent_chunks,
                trim_to_active=input_ids.size(0) == 1,
            )

            latent_trajectory = reasoning_output.latent_trajectory.to(prompt_embeddings.dtype)
            input_embeddings = torch.cat([prompt_embeddings, latent_trajectory], dim=1)
            input_embeddings = input_embeddings.to(self.slow_reasoning_model.dtype)
            input_mask = torch.cat(
                [prompt_mask, reasoning_output.latent_attention_mask.to(prompt_mask.dtype)],
                dim=1,
            ).long()

            outputs = self.slow_reasoning_model.generate(
                inputs_embeds=input_embeddings,
                attention_mask=input_mask,
                generation_config=generation_config,
                **generation_kwargs,
            )

        if return_latent_budget:
            return outputs, {
                "active_chunks": reasoning_output.active_chunks.detach().cpu(),
                "active_tokens": reasoning_output.active_tokens.detach().cpu(),
                "budget_values": reasoning_output.budget_values.detach().cpu(),
            }
        return outputs
