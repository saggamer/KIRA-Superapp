"""Frozen source-only features plus separately encoded complete instructions."""
import numpy as np
import torch
from PIL import Image
from .horizon_grounded_runtime import source_features
from .horizon_relationships import validate_conditioning_budget

@torch.no_grad()
def image_features(pipe,source):
 source=source.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
 z,ids=source_features(pipe,source)
 if not hasattr(pipe,'_horizon_generic_context'):pipe._horizon_generic_context=pipe.encode_prompt(prompt='Describe the visible scene.',device=z.device,max_sequence_length=256)
 generic,text_ids=pipe._horizon_generic_context
 captured=[]
 def capture(module,inputs):captured.append(inputs[0].detach())
 handle=pipe.transformer.proj_out.register_forward_pre_hook(capture)
 if pipe.transformer.horizon_region_hint is not None:raise ValueError('Source feature extraction cannot use a region hint')
 try:
  pipe.transformer(hidden_states=pipe._pack_latents(z),timestep=torch.zeros(1,device=z.device,dtype=z.dtype),guidance=None,encoder_hidden_states=generic,txt_ids=text_ids,img_ids=ids,return_dict=False)
 finally:handle.remove()
 if len(captured)!=1:raise ValueError('Native image feature extraction changed')
 rgb=torch.from_numpy(np.asarray(source).copy()).to(z.device).float()/255
 rgb=torch.nn.functional.adaptive_avg_pool2d(rgb.permute(2,0,1)[None],z.shape[-2:]).flatten(2).transpose(1,2)
 return captured[0],rgb

@torch.no_grad()
def instruction_features(pipe,prompt):
 validate_conditioning_budget(pipe.tokenizer,prompt,256)
 embedding,_=pipe.encode_prompt(prompt=prompt,device='cuda',max_sequence_length=256)
 formatted=pipe.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 encoded=pipe.tokenizer(formatted,return_tensors='pt',padding='max_length',truncation=False,max_length=256)
 mask=encoded['attention_mask'].bool().to(embedding.device)
 if embedding.shape[:2]!=mask.shape:raise ValueError('Native instruction token/mask mismatch')
 return embedding,mask

@torch.no_grad()
def predict_instruction_region(pipe,head,visual,rgb,prompt):
 text,mask=instruction_features(pipe,prompt)
 text=text[:,mask[0]];mask=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool)
 with torch.autocast('cuda',dtype=torch.bfloat16):logits=head(visual,rgb,text,mask)
 return logits.float().sigmoid().transpose(1,2).reshape(1,1,32,32)

def load_instruction_grounder(directory):
 import json
 from pathlib import Path
 from safetensors.torch import load_file
 from .horizon_instruction_grounder import InstructionGrounder
 path=Path(directory);config=json.loads((path/'instruction_grounder.json').read_text())
 head=InstructionGrounder(config['image_width'],config['text_width'],config['width']).to('cuda')
 head.load_state_dict(load_file(str(path/'instruction_grounder.safetensors')),strict=True);head.eval();return head
