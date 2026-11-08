from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description='Horizon 1 experimental MLX image generation and editing for Apple Silicon')
    parser.add_argument('--bundle',type=Path,required=True,help='Folder downloaded from the Horizon 1 model repository')
    parser.add_argument('--base-path',type=Path)
    parser.add_argument('--source',type=Path,help='Source image for editing; omit for text-to-image generation')
    parser.add_argument('--mode',choices=['edit','generate'])
    parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--height',type=int,default=512)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--guidance',type=float,default=4.0)
    parser.add_argument('--constrain-predicted-region',action='store_true')
    parser.add_argument('--protect-predicted-region',action='store_true',
                        help='Save an additional pixel-copy composite, explicitly separate from the raw result')
    args=parser.parse_args()
    mode=args.mode or ('edit' if args.source else 'generate')
    if not args.prompt.strip():parser.error('Provide a nonempty editing instruction')
    if not 0<=args.seed<=2**32-1:parser.error('Seed must be a 32-bit unsigned integer')
    if mode=='edit' and (not args.source or not args.source.is_file()):parser.error('Editing requires a source image that exists')
    if mode=='generate' and args.source:parser.error('Generation does not take a source image; use edit mode')
    if mode=='generate' and (args.constrain_predicted_region or args.protect_predicted_region):parser.error('Region controls apply only to editing')
    if mode=='generate' and any(not 256<=size<=1024 or size%16 for size in [args.width,args.height]):parser.error('Dimensions must be multiples of 16 between 256 and 1024')
    if mode=='edit' and (args.width,args.height)!=(512,512):parser.error('The editor uses a 512 × 512 canvas')
    if args.output.suffix.lower()!='.png':parser.error('Use a .png output filename')
    if not 1 <= args.steps <= 200:parser.error('Steps must be between 1 and 200')
    if not 1 <= args.guidance <= 20:parser.error('Guidance must be between 1 and 20')
    paths=[args.output,args.output.with_suffix('.json')]
    if mode=='edit':paths.extend([args.output.with_suffix('.predicted-mask.png'),args.output.with_suffix('.probability.png')])
    if args.protect_predicted_region:paths.append(args.output.with_suffix('.protected.png'))
    if any(p.exists() for p in paths):parser.error('An output already exists; use a fresh filename')
    from PIL import Image,ImageOps
    from .runtime import HorizonMLX
    model=HorizonMLX(args.bundle,args.base_path,mode=mode)
    if mode=='generate':
        image,metadata=model.generate(args.prompt,args.seed,args.steps,args.guidance,args.width,args.height)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        image.save(args.output)
    else:
        with Image.open(args.source) as source:
            source=ImageOps.exif_transpose(source).convert('RGB')
            image,mask,heatmap,metadata=model.edit(source,args.prompt,args.seed,args.steps,args.guidance,args.constrain_predicted_region)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            image.save(paths[0]);mask.save(paths[2]);heatmap.save(paths[3])
            if args.protect_predicted_region:Image.composite(image,source.resize((512,512),Image.Resampling.LANCZOS),mask).save(paths[-1])
        metadata['protected_output_is_pixel_copy']=args.protect_predicted_region
    hashes=[('output_sha256',args.output)]
    if args.source:hashes.append(('source_sha256',args.source))
    for key,path in hashes:
        with path.open('rb') as f:metadata[key]=hashlib.file_digest(f,'sha256').hexdigest()
    paths[1].write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
