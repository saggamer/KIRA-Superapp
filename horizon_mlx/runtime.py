"""Horizon 1 MLX generation and editing with separate trained visual experts."""
from pathlib import Path
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageOps
from .components import InstructionGrounder, SemanticBridge, constrain_region

BASE_REPO = 'AITRADER/FLUX2-klein-base-4B-mlx-4bit'
BASE_REVISION = 'e04a5bc9307320a601ebb0047e3f47ec7dc4d432'


class _Context(nn.Module):
    def __init__(self, inner, bridge):
        super().__init__()
        self.inner, self.bridge = inner, bridge

    def __call__(self, x):
        return self.inner(self.bridge(x))


class _ImageInput(nn.Module):
    def __init__(self, inner, embedding):
        super().__init__()
        self.inner, self.embedding, self.hint = inner, embedding, None

    def __call__(self, x):
        out = self.inner(x)
        if self.hint is not None:
            if out.shape[1] != 2 * self.hint.shape[1]:
                raise ValueError('Target and source token grids must align with the predicted region')
            out = out + mx.concatenate([self.hint, self.hint], axis=1).astype(out.dtype) * self.embedding.astype(out.dtype)
        return out


class _Output(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.feature_mode = inner, False

    def __call__(self, x):
        return x if self.feature_mode else self.inner(x)


class HorizonMLX:
    """Experimental trained image experts. Loading executes no remote model code."""
    def __init__(self, bundle, base_path=None, mode='edit'):
        from huggingface_hub import snapshot_download
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        import json
        if mode not in {'edit','generate'}:raise ValueError('Use edit or generate mode')
        self.mode=mode
        self.bundle = Path(bundle)
        self.local_base_override = base_path is not None
        self.threshold = json.loads((self.bundle / 'threshold.json').read_text())['threshold']
        weights = self.bundle / 'weights'
        base_path = base_path or snapshot_download(BASE_REPO, revision=BASE_REVISION)
        generation=mode=='generate'
        adapter=weights/('creativity_lora.safetensors' if generation else 'editing_lora.safetensors')
        bridge=weights/('creativity_bridge.safetensors' if generation else 'semantic_bridge.safetensors')
        if not adapter.is_file() or not bridge.is_file():
            raise FileNotFoundError('This Horizon bundle lacks the requested expert. Run ./DOWNLOAD_MODELS.command --horizon-only to update it.')
        print(f'Loading 4-bit base, native Qwen encoder, and Horizon {"creativity" if generation else "editing"} expert...', flush=True)
        model_class=Flux2Klein if generation else Flux2KleinEdit
        self.model = model_class(model_config=ModelConfig.flux2_klein_base_4b(),
                                   model_path=str(base_path), quantize=4, compute_precision=mx.float16,
                                   lora_paths=[str(adapter)],
                                   lora_scales=[1.0], bake_lora=False)
        self.head = None if generation else InstructionGrounder(weights / 'instruction_grounder.safetensors')
        t = self.model.transformer
        t.context_embedder = _Context(t.context_embedder, SemanticBridge(bridge,mode='creativity' if generation else 'editing'))
        if not generation:
            t.x_embedder = _ImageInput(t.x_embedder, mx.load(str(weights / 'region_embedding.safetensors'))['region_embedding'])
            t.proj_out = _Output(t.proj_out)
        mx.eval(self.model.parameters())

    def encode(self, prompt):
        from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
        tokenizer = self.model.tokenizers['qwen3']
        # Count the COMPLETE native chat-template prompt before any padding/truncation.
        formatted = tokenizer.tokenizer.apply_chat_template(
            [{'role':'user', 'content':prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer.tokenizer(formatted, padding='max_length', max_length=256,
                                      truncation=False, return_tensors='np', add_special_tokens=True)
        if encoded['input_ids'].shape[1] > 256:
            raise ValueError('Instruction exceeds the trained 256-token native context; shorten it explicitly')
        ids, mask = mx.array(encoded['input_ids']), mx.array(encoded['attention_mask'])
        embeds = self.model.text_encoder.get_prompt_embeds(ids, mask, hidden_state_layers=(9,18,27))
        mx.eval(embeds)
        if not bool(mx.all(mx.isfinite(embeds))):
            raise FloatingPointError('Nonfinite text encoder output')
        return embeds, Flux2PromptEncoder.prepare_text_ids(embeds), mask

    def edit(self, source, prompt, seed=42, steps=20, guidance=4.0, constrain=False):
        if self.mode!='edit':raise ValueError('Load the editing expert for source-image edits')
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers as Helpers
        from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator as Latents
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler as Scheduler
        from mflux.utils.image_util import ImageUtil
        if not prompt.strip() or not 1 <= steps <= 200 or not 1 <= guidance <= 20:
            raise ValueError('Provide a nonempty instruction, 1–200 steps, and guidance between 1 and 20')
        started = time.monotonic()
        source = ImageOps.exif_transpose(source).convert('RGB').resize((512,512), Image.Resampling.LANCZOS)
        instruction, text_ids, instruction_mask = self.encode(prompt)
        generic, generic_ids, _ = self.encode('Describe the visible scene.')
        negative, negative_ids, _ = self.encode('') if guidance > 1 else (None,None,None)
        print('Encoding source image and predicting the instruction-specific edit region...', flush=True)
        encoded = VAEUtil.encode(vae=self.model.vae, image=ImageUtil.to_array(source), tiling_config=None)
        encoded = Helpers.ensure_4d_latents(encoded)
        packed = Helpers.bn_normalize_vae_encoded_latents(Latents.patchify_latents(encoded), vae=self.model.vae)
        clean = Latents.pack_latents(packed)
        source_ids = Latents.prepare_grid_ids(packed, t_coord=0)
        reference_ids = Latents.prepare_grid_ids(packed, t_coord=10)
        transformer = self.model.transformer
        transformer.proj_out.feature_mode = True
        try:
            visual = transformer(hidden_states=clean, encoder_hidden_states=generic,
                                 timestep=mx.array(0.0), img_ids=source_ids, txt_ids=generic_ids)
            mx.eval(visual)
        finally:
            transformer.proj_out.feature_mode = False
        pixels = np.asarray(source, dtype=np.float32) / 255
        rgb = mx.array(pixels.reshape(32,16,32,16,3).mean((1,3)).reshape(1,1024,3))
        # Grounder saw only non-padding instruction tokens during training.
        positions = mx.array(np.flatnonzero(np.array(instruction_mask)[0]), dtype=mx.int32)
        text = mx.take(instruction, positions, axis=1)
        probability = mx.sigmoid(self.head(visual, rgb, text, mx.ones(text.shape[:2], dtype=mx.bool_)))
        mx.eval(probability)
        if not bool(mx.all(mx.isfinite(probability))):
            raise FloatingPointError('Nonfinite grounding probability')
        region = (probability > self.threshold).astype(mx.float32)
        del visual, generic, encoded, packed
        mx.clear_cache()
        latents, latent_ids, lh, lw = Helpers.prepare_generation_latents(seed=seed,height=512,width=512)
        initial_noise = latents
        timesteps, sigmas = Scheduler.get_timesteps_and_sigmas(image_seq_len=1024,num_inference_steps=steps)
        transformer.x_embedder.hint = region
        # Use MFLUX's native predictor: compiled on supported Apple GPUs, eager on M1/M2.
        # Construct a new closure after setting this image's hint so a compiled graph cannot
        # retain another image's predicted region. Feature extraction above remains eager.
        predict = self.model._predict(transformer)
        try:
            for i in range(steps):
                prediction = predict(latents=latents,image_latents=clean,latent_ids=latent_ids,
                                     image_latent_ids=reference_ids,prompt_embeds=instruction,text_ids=text_ids,
                                     negative_prompt_embeds=negative,negative_text_ids=negative_ids,
                                     guidance=guidance,timestep=timesteps[i])
                latents = (latents.astype(mx.float32) + (sigmas[i+1]-sigmas[i])*prediction.astype(mx.float32)).astype(initial_noise.dtype)
                if constrain:
                    latents = constrain_region(latents,clean,initial_noise,region,sigmas[i+1])
                mx.eval(latents)
                if not bool(mx.all(mx.isfinite(latents))):
                    raise FloatingPointError(f'Nonfinite latents at step {i+1}')
                print(f'Edit step {i+1}/{steps}', flush=True)
        finally:
            transformer.x_embedder.hint = None
            del predict
        decoded = self.model.vae.decode_packed_latents(latents.reshape(1,lh,lw,128).transpose(0,3,1,2))
        mx.eval(decoded)
        if not bool(mx.all(mx.isfinite(decoded))):
            raise FloatingPointError('Nonfinite decoder output')
        image = ImageUtil.to_pil(decoded)
        values = np.array(probability).reshape(32,32)
        mask = Image.fromarray(((values > self.threshold)*255).astype('uint8')).resize((512,512),Image.Resampling.NEAREST)
        heatmap = Image.fromarray(np.clip(values*255,0,255).astype('uint8')).resize((512,512),Image.Resampling.NEAREST)
        metadata = {'model':'Horizon 1', 'backend':'MLX/MFLUX', 'base_repo':BASE_REPO,
                    'base_revision':BASE_REVISION, 'base_bits':4,'auxiliary_precision':'float32',
                    'local_base_override':self.local_base_override,
                    'base_download_pinned':not self.local_base_override,
                    'prompt':prompt,'effective_prompt':prompt,'seed':seed,'steps':steps,'guidance':guidance,
                    'working_resolution':[512,512],'predicted_token_fraction':float(mx.mean(region)),
                    'threshold':self.threshold,'latent_region_constraint':constrain,
                    'inference_inputs':['source_image','original_instruction'],
                    'target_or_annotation_supplied':False,'semantic_success':None,
                    'seconds':time.monotonic()-started,'promoted':False}
        return image, mask, heatmap, metadata

    def generate(self, prompt, seed=42, steps=20, guidance=4.0, width=512, height=512):
        """Text-to-image through the trained creativity expert; no image conditioning."""
        from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
        from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
        from mflux.utils.image_util import ImageUtil
        if self.mode!='generate':raise ValueError('Load the creativity expert for text-to-image generation')
        if not prompt.strip() or not 1<=steps<=200 or not 1<=guidance<=20:
            raise ValueError('Provide a nonempty prompt, 1–200 steps, and guidance between 1 and 20')
        if any(not 256<=size<=1024 or size%16 for size in [width,height]):
            raise ValueError('Image dimensions must be multiples of 16 between 256 and 1024')
        started=time.monotonic()
        context,text_ids,_=self.encode(prompt)
        negative,negative_ids,_=self.encode('') if guidance>1 else (None,None,None)
        latents,latent_ids,lh,lw=Flux2LatentCreator.prepare_packed_latents(seed=seed,height=height,width=width,batch_size=1)
        timesteps,sigmas=FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
            image_seq_len=latents.shape[1],num_inference_steps=steps)
        predict=self.model._predict(self.model.transformer)
        try:
            for i in range(steps):
                prediction=predict(latents=latents,latent_ids=latent_ids,prompt_embeds=context,text_ids=text_ids,
                                   negative_prompt_embeds=negative,negative_text_ids=negative_ids,
                                   guidance=guidance,timestep=timesteps[i])
                latents=(latents.astype(mx.float32)+(sigmas[i+1]-sigmas[i])*prediction.astype(mx.float32)).astype(latents.dtype)
                mx.eval(latents)
                if not bool(mx.all(mx.isfinite(latents))):raise FloatingPointError(f'Nonfinite generation latents at step {i+1}')
                print(f'Generate step {i+1}/{steps}',flush=True)
        finally:del predict
        decoded=self.model.vae.decode_packed_latents(latents.reshape(1,lh,lw,128).transpose(0,3,1,2))
        mx.eval(decoded)
        if not bool(mx.all(mx.isfinite(decoded))):raise FloatingPointError('Nonfinite generated image')
        image=ImageUtil.to_pil(decoded)
        return image,{'model':'Horizon 1','mode':'text-to-image','expert':'creativity',
                     'backend':'MLX/MFLUX','base_repo':BASE_REPO,'base_revision':BASE_REVISION,'base_bits':4,
                     'prompt':prompt,'effective_prompt':prompt,'seed':seed,'steps':steps,'guidance':guidance,
                     'working_resolution':[width,height],'source_image_supplied':False,'reference_image_tokens':0,
                     'grounding_head_used':False,'region_hint_used':False,'semantic_success':None,
                     'new_training_performed':False,'seconds':time.monotonic()-started,'promoted':False}
