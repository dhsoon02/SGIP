#!/usr/bin/env python3
"""Generate the 72-second SGIP overview from the original manuscript ZIP.

Requirements: Python 3.10+, Pillow, PyMuPDF, and FFmpeg with libx264.
Example:
    python render_overview_portable.py --paper manuscript.zip --output overview.mp4
Add --stills-only to validate layouts without encoding a video.
The output is an editorial animation of paper figures, not motion playback.
"""
from pathlib import Path, PurePosixPath
import argparse
import shutil
import subprocess
import tempfile
import zipfile

from PIL import Image, ImageDraw, ImageFont
import pymupdf

FIGURES = ('final_teaser', 'final_main', 'final_qual', 'final_mult')


def render_paper_figures(paper, work):
    """Read only the four requested PDFs; no ZIP paths are extracted to disk."""
    assets = {}
    with zipfile.ZipFile(paper) as archive:
        for name in FIGURES:
            suffix = ('figures', name + '.pdf')
            matches = [member for member in archive.namelist()
                       if tuple(PurePosixPath(member).parts[-2:]) == suffix]
            if len(matches) != 1:
                raise ValueError(
                    f'Expected one figures/{name}.pdf in {paper}; found {len(matches)}.'
                )
            with pymupdf.open(stream=archive.read(matches[0]), filetype='pdf') as doc:
                if len(doc) == 0:
                    raise ValueError(f'{matches[0]} contains no PDF pages.')
                page = doc[0]
                scale = max(2, 2400 / page.rect.width)
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale), alpha=False
                )
                figure_path = work / (name + '.png')
                pixmap.save(figure_path)
            with Image.open(figure_path) as image:
                assets[name] = image.convert('RGB')
    return assets


def render(args, work):
    W,H,FPS=1920,1080,24
    BG='#F6F8F7'; INK='#122A3D'; MUTED='#526772'; TEAL='#007F82'; PALE='#E2F2EE'; GOLD='#E8B84B'; ORANGE='#DA7A32'; BORDER='#DCE5E2'
    REG=args.font_dir/'DejaVuSans.ttf'; BOLD=args.font_dir/'DejaVuSans-Bold.ttf'
    fonts={}
    def f(size,bold=False):
     key=(size,bold)
     if key not in fonts: fonts[key]=ImageFont.truetype(str(BOLD if bold else REG),size)
     return fonts[key]
    def txt(im,text,x,y,size=28,color=INK,bold=False,spacing=12):
     d=ImageDraw.Draw(im)
     d.multiline_text((x,y),text,font=f(size,bold),fill=color,spacing=spacing,stroke_width=0)
    def roundbox(im,box,fill='white',outline=BORDER,r=22,width=2):
     ImageDraw.Draw(im).rounded_rectangle(tuple(int(v) for v in box),radius=r,fill=fill,outline=outline,width=width)
    def tag(im,text,x,y,color=TEAL,fill=PALE,size=23):
     d=ImageDraw.Draw(im); b=d.textbbox((0,0),text,font=f(size,True)); width=b[2]+38
     roundbox(im,(x,y,x+width,y+46),fill,fill,r=23,width=0);txt(im,text,x+19,y+8,size,color,True)
     return width
    assets=render_paper_figures(args.paper, work)
    def crop(name,box=None):
     im=assets[name]
     if box:
      w,h=im.size;return im.crop(tuple(round(v*(w if j%2==0 else h)) for j,v in enumerate(box)))
     return im.copy()
    def fig(im,asset,box,padding=20):
     x,y,w,h=box;roundbox(im,(x,y,x+w,y+h)); a=asset.copy(); a.thumbnail((int(w-2*padding),int(h-2*padding)),Image.Resampling.LANCZOS)
     px=round(x+(w-a.width)/2);py=round(y+(h-a.height)/2);im.paste(a,(px,py));return (x,y,x+w,y+h)
    def base(chapter=None,num=0):
     im=Image.new('RGB',(W,H),BG)
     txt(im,'SGIP',100,51,32,INK,True)
     ImageDraw.Draw(im).line((216,57,216,92),fill=BORDER,width=2)
     txt(im,'RESEARCH OVERVIEW',246,59,21,MUTED,True)
     if chapter:txt(im,f'{num:02d}  /  {chapter.upper()}',1280,61,20,TEAL,True)
     ImageDraw.Draw(im).line((100,125,1820,125),fill=BORDER,width=2)
     txt(im,'NeurIPS 2026',100,1013,21,MUTED)
     txt(im,'Original paper figures · Editorial animation',1155,1013,21,MUTED)
     return im
    scenes=[]
    def scene(duration,im,highlights=[],reveals=[]):
     scenes.append({'duration':duration,'image':im,'highlights':highlights,'reveals':reveals})
    # 00. A compact title and the full original teaser.
    im=base('Learning to ground future motion',1)
    tag(im,'NeurIPS 2026',100,180)
    txt(im,'SGIP',94,253,138,INK,True)
    txt(im,'Learning Scene-Grounded\nInteraction Priors for\nScene-Aware Human\nMotion Prediction',102,443,37,INK,True,16)
    txt(im,'Where will we interact?\nHow will we move?',104,718,29,MUTED,False,10)
    box=fig(im,crop('final_teaser'),(820,270,1000,510),20)
    tag(im,'OBSERVED MOTION + 3D SCENE',845,815,size=21)
    txt(im,'A scene-grounded prior for future human motion.',845,888,23,MUTED)
    scene(7,im,[(box,1.5,TEAL)])
    # 01. Ambiguity.
    im=base('The challenge',2)
    txt(im,'One observed motion. Several possible futures.',100,177,49,INK,True)
    txt(im,'A person can approach multiple plausible interaction targets.',103,253,27,MUTED)
    box=fig(im,crop('final_teaser',(0,0,1/3,1)),(130,337,630,585),20)
    txt(im,'Where should the future be grounded?',850,365,34,INK,True)
    rev=[]
    for t,y,n,body in [(0.7,460,'01','Choose an interaction target.'),(2.0,566,'02','Resolve ambiguity using the scene.'),(3.4,672,'03','Predict the trajectory and body pose.')]:
     layer=Image.new('RGBA',(W,H),(0,0,0,0));tag(layer,n,850,y,size=22);txt(layer,body,940,y+6,27,INK);rev.append((t,layer))
    txt(im,'Objects and free space can support different futures.',850,833,25,MUTED)
    scene(8,im,[(box,0.5,TEAL)],rev)
    # 02. Prior.
    im=base('Scene-grounded interaction prior',3)
    txt(im,'Make the interaction target explicit.',100,177,54,INK,True)
    txt(im,'The student predicts a probability distribution over grounding tokens.',103,257,27,MUTED)
    box=fig(im,crop('final_teaser',(1/3,0,2/3,1)),(130,337,630,585),20)
    txt(im,'Grounding tokens',850,365,38,INK,True)
    rev=[]
    for t,y,head,body in [(0.6,462,'Object instances','Specific chairs, sofas, bags, and other objects.'),(2.2,592,'Navigation-region token','A scene-global option for object-agnostic motion.'),(4.0,722,'Prior-conditioned prediction','The selected token guides the future motion.')]:
     layer=Image.new('RGBA',(W,H),(0,0,0,0));txt(layer,head,850,y,29,TEAL,True);txt(layer,body,850,y+49,24,MUTED);rev.append((t,layer))
    scene(8,im,[(box,.5,TEAL)],rev)
    # 03. Training only, full pipeline preserved and legible.
    im=base('Privileged learning',4)
    txt(im,'Intent helps during training.',100,177,54,INK,True)
    tag(im,'TRAINING ONLY',100,262,ORANGE,'#FBEBDD')
    txt(im,'Textual intent + semantic and geometric scores form the teacher prior.',395,270,26,MUTED)
    box=fig(im,crop('final_main',(0,0,0.665,0.47)),(100,349,1720,485),22)
    # Below-figure narration makes the distilled quantities explicit.
    roundbox(im,(100,864,1820,963),PALE,PALE,r=18,width=0)
    txt(im,'Teacher → Student',130,891,30,TEAL,True)
    txt(im,'Distill both the grounding prior and the predicted motion.',520,895,27,INK)
    scene(10,im,[(box,1,ORANGE)])
    # 04. Inference.
    im=base('Observation-only inference',5)
    txt(im,'At inference: motion + scene only.',100,177,54,INK,True)
    tag(im,'NO GAZE',100,262);tag(im,'NO TEXTUAL INTENT',289,262);tag(im,'NO LLM',610,262)
    box=fig(im,crop('final_main',(0,0.595,0.665,1)),(100,350,1720,485),22)
    roundbox(im,(100,864,1820,963),PALE,PALE,r=18,width=0)
    txt(im,'Student prior → Selected token → Future trajectory + full-body pose',130,897,31,INK,True)
    scene(8,im,[(box,0.6,TEAL)])
    # 05 and 06. Original comparison panels with complete legend in page.
    for j,(label,region) in enumerate([('Kitchen',(0,0,0.332,0.916)),('Living room',(.335,0,.665,.916))]):
     im=base('Qualitative comparison',6+j)
     txt(im,'Scene-aware motion, shown in the paper.',100,177,49,INK,True)
     txt(im,'Same scene · SGIP versus gaze-free SIF3D*',103,253,27,MUTED)
     box=fig(im,crop('final_qual',region),(130,324,575,652),10)
     tag(im,label.upper(),815,365)
     txt(im,'Compare the predicted path\nand final interaction.',815,453,40,INK,True,14)
     txt(im,'Top: SGIP\nBottom: SIF3D* without gaze',818,601,29,MUTED,False,18)
     for y,color,text in [(759,'#D6AA13','Observed motion'),(817,'#B62F4F','Ground-truth future'),(875,'#1764A6','Predicted future')]:
      ImageDraw.Draw(im).ellipse((821,y+8,841,y+28),fill=color);txt(im,text,865,y,25,MUTED)
     scene(5,im,[(box,0.5,TEAL)])
    # 07. Multiple futures.
    im=base('Token-conditioned alternatives',8)
    txt(im,'Change the token. Change the future.',100,177,54,INK,True)
    txt(im,'Keep the observed motion, scene, and model parameters fixed.',103,257,27,MUTED)
    box=fig(im,crop('final_mult'),(100,366,1720,485),20)
    txt(im,'Ground-truth interaction object',150,892,29,INK,True)
    txt(im,'Alternative interaction object',1020,892,29,INK,True)
    scene(8,im,[((100,366,957,851),0.7,TEAL),((963,366,1820,851),3.8,ORANGE)])
    # 08. Precisely scoped quantitative results.
    im=base('Benchmark results',9)
    txt(im,'Lower trajectory errors.',100,177,60,INK,True)
    txt(im,'Comparisons with the indicated gaze-free baselines. Lower is better.',103,263,27,MUTED)
    roundbox(im,(100,355,935,876));roundbox(im,(975,355,1820,876))
    for x,dataset,metric,pct,baseline,old,new in [(140,'GIMO','Traj-path','14.5%','GAP3DS* · no gaze',642,549),(1015,'PROX','Traj-dest','12.6%','SIF3D* · no gaze',659,576)]:
     tag(im,dataset,x,394)
     txt(im,metric,x,470,32,INK,True)
     txt(im,pct,x,525,104,TEAL,True)
     txt(im,'lower error',x+5,653,28,TEAL)
     txt(im,f'{baseline}: {old}',x,732,25,MUTED)
     txt(im,f'SGIP: {new}',x,784,30,INK,True)
    txt(im,'Relative reduction: (baseline − SGIP) / baseline. Values reported in the paper.',103,921,25,MUTED)
    scene(7,im,[((100,355,935,876),0.6,TEAL),((975,355,1820,876),2.6,TEAL)])
    # 09. Closing.
    im=base('Explore the research',10)
    tag(im,'NeurIPS 2026',100,188)
    txt(im,'Ground the interaction.\nPredict the motion.',100,296,79,INK,True,24)
    ImageDraw.Draw(im).rounded_rectangle((105,545,541,552),radius=3,fill=TEAL)
    txt(im,'SGIP',97,611,100,TEAL,True)
    txt(im,'Learning Scene-Grounded Interaction Priors\nfor Scene-Aware Human Motion Prediction',105,750,32,INK,False,13)
    txt(im,'dhsoon02.github.io/SGIP/',105,900,34,TEAL,True)
    # Include a faithful final predicted-motion panel, without animating the human poses.
    fig(im,crop('final_teaser',(2/3,0,1,1)),(1240,270,580,550),18)
    scene(6,im)

    CAPTIONS=[
     (0,7,'SGIP: Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction. NeurIPS 2026.'),
     (7,15,'The same observed motion can lead to several plausible interaction targets. SGIP makes future-motion grounding explicit.'),
     (15,23,'The student predicts a grounding prior over object instances and a navigation-region token, then conditions motion on the selected token.'),
     (23,28,'During training, textual intent and semantic and geometric scores construct a privileged teacher prior.'),
     (28,33,'Both the grounding prior and predicted motion are distilled into an observation-only student.'),
     (33,41,'At inference, SGIP uses observed motion and a 3D scene only. It requires no gaze, textual intent, or LLM.'),
     (41,46,'Kitchen comparison from the paper. SGIP is shown above gaze-free SIF3D*. Yellow: observed. Red: ground-truth future. Blue: predicted future.'),
     (46,51,'Living-room comparison from the paper. Compare the predicted trajectory and final interaction under the same scene.'),
     (51,59,'Changing the selected interaction token changes the predicted path and final pose, while keeping the observed motion, scene, and model fixed.'),
     (59,66,'GIMO Traj-path error is 14.5 percent lower than gaze-free GAP3DS*, 642 to 549. PROX Traj-dest error is 12.6 percent lower than gaze-free SIF3D*, 659 to 576.'),
     (66,72,'Ground the interaction. Predict the motion. Explore SGIP at dhsoon02.github.io/SGIP/.')]

    def smooth(x):return max(0,min(1,x))**2*(3-2*max(0,min(1,x)))
    def frame_for(i,local):
     s=scenes[i];im=s['image'].copy().convert('RGBA')
     # Editorial text reveals are placed outside the scientific figure pixels.
     for when,layer in s['reveals']:
      a=smooth((local-when)/.65)
      if a>0:
       layer=layer.copy();layer.putalpha(layer.getchannel('A').point(lambda p:int(p*a)));im.alpha_composite(layer,(0,round((1-a)*14)))
     # Progressively draw a subtle outline around a figure/panel.
     d=ImageDraw.Draw(im)
     for box,when,color in s['highlights']:
      a=smooth((local-when)/.8)
      if a>0:
       x1,y1,x2,y2=box
       col=tuple(int(color[k:k+2],16) for k in (1,3,5))+(int(a*195),)
       overlay=Image.new('RGBA',(W,H),(0,0,0,0));od=ImageDraw.Draw(overlay)
       od.rounded_rectangle((x1-1,y1-1,x2+1,y2+1),radius=22,outline=col,width=3)
       im=Image.alpha_composite(im,overlay)
     return im.convert('RGB')
    def timestamp(t):return f'{int(t)//3600:02d}:{int(t)%3600//60:02d}:{int(t)%60:02d}.000'
    def main():
     out=args.output;out.parent.mkdir(parents=True,exist_ok=True)
     starts=[];total=0
     for s in scenes:starts.append(total);total+=s['duration']
     assert total==72
     with open(out.with_suffix('.en.vtt'),'w',encoding='utf-8') as fp:
      fp.write('WEBVTT\n\n')
      for a,b,c in CAPTIONS:fp.write(f'{timestamp(a)} --> {timestamp(b)}\n{c}\n\n')
     for i in range(len(scenes)):
      frame_for(i,scenes[i]['duration']-.8).save(out.parent/f'scene-{i+1:02d}.jpg',quality=92)
     frame_for(0,3).save(out.parent/'sgip-overview-poster.jpg',quality=95)
     # Contact sheet for validating text, composition, and figure provenance.
     contact=Image.new('RGB',(960,310*5),BG)
     for i in range(len(scenes)):
      mini=frame_for(i,scenes[i]['duration']-.8);mini.thumbnail((480,270));contact.paste(mini,((i%2)*480,(i//2)*310))
      ImageDraw.Draw(contact).text(((i%2)*480+10,(i//2)*310+284),f'{starts[i]:02d}s · scene {i+1}',font=f(18),fill=INK)
     contact.save(out.parent/'contact-sheet.jpg',quality=94)
     if args.stills_only:return
     cmd=[shutil.which('ffmpeg'),'-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-metadata','title=SGIP | Research overview','-metadata','comment=Editorial overview of original paper figures. Not motion-sequence playback.',str(out)]
     proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
     for n in range(total*FPS):
      t=n/FPS;i=max(j for j,a in enumerate(starts) if a<=t);local=t-starts[i];im=frame_for(i,local)
      # Crossfades preserve the duration and avoid interrupting the storyline.
      if i and local<.58:
       prev=frame_for(i-1,scenes[i-1]['duration']-.1);im=Image.blend(prev,im,smooth(local/.58))
      d=ImageDraw.Draw(im);d.rectangle((0,H-7,int(W*t/total),H),fill=TEAL)
      proc.stdin.write(im.tobytes())
      if n%(FPS*6)==0:print(f'Rendered {t:.0f}/{total}s',flush=True)
     proc.stdin.close();code=proc.wait()
     if code:raise RuntimeError(f'ffmpeg exited {code}')
     print(f'Wrote {out} ({out.stat().st_size/1024/1024:.1f} MiB)',flush=True)
    main()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True,
                        help='Manuscript ZIP containing figures/final_*.pdf.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Destination MP4; related stills and captions use its directory.')
    parser.add_argument('--font-dir', type=Path,
                        default=Path('/usr/share/fonts/truetype/dejavu'),
                        help='Directory containing DejaVuSans.ttf and DejaVuSans-Bold.ttf.')
    parser.add_argument('--stills-only', action='store_true',
                        help='Render ten chapter stills, poster, contact sheet, and captions only.')
    args = parser.parse_args()
    if not args.paper.is_file():
        parser.error(f'Manuscript ZIP does not exist: {args.paper}')
    if not zipfile.is_zipfile(args.paper):
        parser.error(f'Not a valid ZIP archive: {args.paper}')
    for name in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'):
        if not (args.font_dir / name).is_file():
            parser.error(f'Missing {args.font_dir / name}; specify --font-dir.')
    if not args.stills_only and shutil.which('ffmpeg') is None:
        parser.error('FFmpeg was not found on PATH; install it with libx264 support.')
    return args


def main():
    args = parse_arguments()
    with tempfile.TemporaryDirectory(prefix='sgip-overview-') as work:
        render(args, Path(work))


if __name__ == '__main__':
    main()
