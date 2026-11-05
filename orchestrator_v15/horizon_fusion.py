"""Dependency-free architecture contract for Horizon's visual training islands."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HorizonFusionArchitecture:
    name: str = 'Horizon 1'
    architecture_revision: int = 2
    visual_donor: str = 'black-forest-labs/FLUX.2-klein-base-4B'
    visual_modes: tuple = ('editing', 'creativity')
    bridge_rank: int = 32
    lora_rank: int = 8
    controller_changed: bool = False
    fully_unified_bagel_model: bool = False

    def relationship_context(self):
        return {'implemented':['object/part scene graphs','scoped materials and counts',
                               'graph-conditioned research prompt compilation','explicit relationship review',
                               'generalized attention reference mask','mode-gated joint-context residual prototype',
                               'role-sensitive relationship scoring prototype'],
                'neural_status':'new shared-context and relationship heads are untrained and not attached to saved image adapters',
                'attention':'full token sequence; causal text, bidirectional visual splits, sample isolation, no noisy-target reuse',
                'native_bridge_checkpoint_compatible':True,
                'prototype_additional_trainable_per_block':36*4096*32+6,
                'eight_block_design_allowance':8*(36*4096*32+6),
                'parameter_scope':'Residuals only; reuses a compatible frozen full-width QKV projection. Native donor conversion remains unimplemented.'}

    def route(self, mode):
        if mode in self.visual_modes:
            return {'island': 'visual', 'adapter': mode, 'bridge': mode,
                    'relationship_graph_supported':True,'joint_context_prototype_enabled':False}
        if mode in {'conversation', 'tools', 'ppt', 'pdf'}:
            return {'island': 'existing_controller', 'adapter': None, 'bridge': None}
        raise ValueError('Unknown Horizon mode: ' + str(mode))

    def compact_target_budget(self):
        # A future shared-Gemma layout, NOT the existing deployed MoE or pilot.
        parts = {'shared_language_upper_bound': 8_000_000_000,
                 'image_transformer_bound': 4_000_000_000,
                 'separate_vision_encoder_allowance': 450_000_000,
                 'vae_allowance': 100_000_000,
                 'residual_experts_and_bridges_allowance': 80_000_000}
        return {'status': 'design estimate; shared-Gemma conversion not trained',
                'components': parts, 'total_upper_bound': sum(parts.values()),
                'ceiling': 13_000_000_000,
                'requires': 'Replace cloned FFN expert banks with residual adapters and distill Gemma-to-image conditioning; remove the extra Qwen text encoder at deployment.'}

    def to_dict(self):
        return {**asdict(self), 'compact_target': self.compact_target_budget(),
                'relationship_context':self.relationship_context(),
                'pilot_scope': 'Native pretrained Qwen3 conditioning plus visual transformer LoRA and a trainable gated semantic bridge; no full language-backbone or VAE update.'}
