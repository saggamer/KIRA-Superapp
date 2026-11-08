"""Bounded offline subprocess adapter for Horizon 1 image edits in Kira."""
import os
import re
from pathlib import Path
import subprocess
import uuid
from horizon_mlx.catalog import paths, validate_bundle, validate_base
from horizon_mlx.install import runtime_python


def edit_image(prompt, source, workspace, app_root, *, mask=None, region=None, strength=None,
               seed=42, timeout=1200):
    from orchestrator_v15.z_image_tool import _load_raster
    prompt=str(prompt or '').strip()
    if not prompt or len(prompt)>600:raise ValueError('The editing instruction must contain 1–600 characters')
    if mask or region or strength is not None:
        raise ValueError('Horizon predicts its edit region from the source and instruction; omit MASK, REGION and STRENGTH')
    source=Path(source).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:
        raise ValueError('PATH must name a local PNG, JPEG, or WebP image')
    if source.stat().st_size>20*1024*1024:raise ValueError('Source image exceeds 20 MiB')
    _load_raster(source)
    seed=int(seed)
    if not 0<=seed<=2**32-1:raise ValueError('SEED must be a 32-bit unsigned integer')
    bundle,base=paths(app_root)
    try:
        validate_bundle(bundle)
        validate_base(base,hashes=False)
    except (OSError,ValueError,KeyError) as error:
        raise FileNotFoundError('Horizon 1 is not installed completely. Run ./DOWNLOAD_MODELS.command --horizon-only') from error
    executable=runtime_python(app_root)
    if not executable.is_file():raise FileNotFoundError('Run ./DOWNLOAD_MODELS.command --horizon-only to set up Horizon')
    output_dir=Path(workspace)/'generated_images';output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/f'horizon_1_edit_{uuid.uuid4().hex[:12]}.png'
    command=[str(executable),'-m','horizon_mlx','--bundle',str(bundle.resolve()),'--base-path',str(base.resolve()),
             '--source',str(source),'--prompt',prompt,'--output',str(output.resolve()),
             '--seed',str(seed),'--steps','20','--guidance','4','--constrain-predicted-region']
    env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    result=subprocess.run(command,cwd=str(app_root),env=env,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError((result.stderr or result.stdout).strip()[-1600:])
    if not output.is_file():raise RuntimeError('Horizon returned without an edited image')
    image=_load_raster(output)
    if image.size!=(512,512):raise RuntimeError('Horizon returned an invalid working resolution')
    return output


def generate_image(prompt, workspace, app_root, *, seed=42, timeout=1200):
    from orchestrator_v15.z_image_tool import _load_raster
    prompt=str(prompt or '').strip()
    if not prompt or len(prompt)>600:raise ValueError('The image prompt must contain 1–600 characters')
    seed=int(seed)
    if not 0<=seed<=2**32-1:raise ValueError('SEED must be a 32-bit unsigned integer')
    bundle,base=paths(app_root)
    try:
        validate_bundle(bundle);validate_base(base,hashes=False)
    except (OSError,ValueError,KeyError) as error:
        raise FileNotFoundError('Update Horizon with ./DOWNLOAD_MODELS.command --horizon-only to install its creativity expert') from error
    executable=runtime_python(app_root)
    if not executable.is_file():raise FileNotFoundError('The Horizon image runtime is missing; run --horizon-only setup')
    folder=Path(workspace)/'generated_images';folder.mkdir(parents=True,exist_ok=True)
    output=folder/f'horizon_1_generate_{uuid.uuid4().hex[:12]}.png'
    command=[str(executable),'-m','horizon_mlx','--mode','generate','--bundle',str(bundle.resolve()),
             '--base-path',str(base.resolve()),'--prompt',prompt,'--output',str(output.resolve()),
             '--seed',str(seed),'--steps','20','--guidance','4']
    env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    result=subprocess.run(command,cwd=str(app_root),env=env,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError((result.stderr or result.stdout).strip()[-1600:])
    if not output.is_file() or _load_raster(output).size!=(512,512):raise RuntimeError('Horizon did not return a valid generated image')
    return output


def handle_direct_generate(app, request, attachments, chat_id, user_chat_prompt):
    if any(Path(str(item.get('path',''))).suffix.lower() in {'.png','.jpg','.jpeg','.webp'} for item in attachments):return False
    if app._extract_image_sources_from_prompt(request):return False
    if re.search(r'\b(?:pdf|docx|pptx?|presentation|slides?|document|video|animation|delete|rename|upload|send)\b',request,re.I):return False
    explicit=re.search(r'\b(?:generat\w*|create|draw|paint|render|imagine|make)\b',request,re.I)
    image=re.search(r'\b(?:images?|pictures?|photos?|photographs?|illustrations?|paintings?|artwork|scene|render)\b',request,re.I)
    if not explicit and re.match(r'\s*(?:what|who|why|how|where|when|is|are|does|can|could)\b',request,re.I):return False
    if re.fullmatch(r'\s*(?:hello|hi|hey|thanks|thank you)[.!\s]*',request,re.I):return False
    if not image and re.search(r'\b(?:code|script|website|app|folder|terminal|shell)\b',request,re.I):return False
    if re.search(r'\b(?:edit|modify|retouch|recolor|brighter|darker)\b',request,re.I):return False
    app._append_chat_message(chat_id,'user',user_chat_prompt,'User')
    app.response_queue.put({'type':'status','content':'Horizon 1 is creating your image. This may take several minutes.','chat_id':chat_id})
    try:
        app._stop_model_worker()
        output=generate_image(request,app.agentic_workspace,app.app_root)
        app._record_artifact(str(output),kind='image',title='Horizon 1 generated image')
        answer=(f'Generated image: `{output}`\nWorking size: 512 × 512\n\n'
                'Created with Horizon’s creativity expert. This is experimental; inspect counts, placement, and requested details.')
    except Exception as error:answer=f'Horizon 1 could not generate this image: {error}'
    app._append_chat_message(chat_id,'assistant',answer,'Horizon 1')
    app.response_queue.put({'type':'message','content':answer,'model_used':'Horizon 1','chat_id':chat_id})
    return True


def handle_direct_edit(app, request, attachments, chat_id, user_chat_prompt):
    """The selected image model receives the user's actual instruction, without a chat-model rewrite."""
    if re.search(r'\b(?:pdf|docx|pptx?|presentation|slides?|document|video|animation|delete|rename|upload|send)\b',request,re.I):
        return False
    if not re.search(r'\b(?:edit|modify|retouch|recolor|add|put|remove|replace|change|make|have|let|turn|erase|extend|place|insert|swap|generate|create|draw|paint|render|imagine)\b',request,re.I):
        return False
    sources=[str(item['path']) for item in attachments if Path(str(item.get('path',''))).suffix.lower() in {'.png','.jpg','.jpeg','.webp'}]
    sources+=app._extract_image_sources_from_prompt(request)
    sources=list(dict.fromkeys(sources))
    app._append_chat_message(chat_id,'user',user_chat_prompt,'User')
    if len(sources)!=1:
        answer='Horizon 1 edits existing images. Attach one source photo and describe the change.'
    else:
        app.response_queue.put({'type':'status','content':'Horizon 1 is editing your image. This may take several minutes.','chat_id':chat_id})
        try:
            app._stop_model_worker()
            path=edit_image(request,sources[0],app.agentic_workspace,app.app_root)
            app._record_artifact(str(path),kind='image',title='Horizon 1 edit')
            answer=(f'Edited image: `{path}`\nPredicted edit region: `{path.with_suffix(".predicted-mask.png")}`\nWorking size: 512 × 512\n\n'
                    'Horizon 1 is experimental. Check the requested objects, counts, and placement; this render is not a verified success.')
        except Exception as error:
            answer=f'Horizon 1 could not complete this edit: {error}'
    app._append_chat_message(chat_id,'assistant',answer,'Horizon 1')
    app.response_queue.put({'type':'message','content':answer,'model_used':'Horizon 1','chat_id':chat_id})
    return True
