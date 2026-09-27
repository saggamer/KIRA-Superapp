"""Measured offline room-echo and double-talk checks; not a real-room claim."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kira_live.playback_echo import DuplexEchoGuard

def rms(x): return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))

def main():
    rate=16000; seconds=12; n=rate*seconds
    rng=np.random.default_rng(41)
    t=np.arange(n)/rate
    # A voiced, time-varying signal plus shaped broadband energy.
    far=(sum(np.sin(2*np.pi*(f*t+3*np.sin(t*.7))) / j for j,f in enumerate([130,260,390,520,650],1))*.04
        +np.convolve(rng.normal(0,.04,n),np.ones(5)/5,mode='same'))
    envelope=.35+.65*np.sin(t*4)**2
    far=(far*envelope).astype(np.float32)
    reference=ROOT/'kira_voice_tmp/complete_bundle_release_voice/kira_integrated_voice.wav'
    if reference.is_file():
        audio,sr=sf.read(reference,dtype='float32')
        audio=np.interp(np.arange(round(len(audio)*rate/sr))*sr/rate,np.arange(len(audio)),audio)
        far=np.tile(audio,int(np.ceil(n/len(audio))))[:n].astype(np.float32)*.7
    echo=np.zeros(n,dtype=np.float32)
    for delay,gain in [(110,.38),(129,.14),(181,.08),(245,.045)]:
        shift=round(delay*rate/1000); echo[shift:]+=far[:-shift]*gain
    user=(.06*np.sin(2*np.pi*211*t)+.027*np.sin(2*np.pi*422*t)+.014*np.sin(2*np.pi*633*t))*(.5+.5*np.sin(t*6)**2)
    user[:rate*8]=0; user[rate*11:]=0
    guard=DuplexEchoGuard(); cleaned=[]; timings=[]
    for start in range(0,n,320):
        clock=start/rate
        guard.add_playback(far[start:start+320],rate,now=clock)
        tick=time.perf_counter()
        cleaned.append(guard.clean(echo[start:start+320]+user[start:start+320],now=clock+.02))
        timings.append(time.perf_counter()-tick)
    cleaned=np.concatenate(cleaned)
    region=slice(rate*4,rate*8)
    erle=20*np.log10(max(rms(echo[region]),1e-9)/max(rms(cleaned[region]),1e-9))
    double=slice(rate*9,rate*11)
    speech_ratio=rms(cleaned[double])/max(rms(user[double]),1e-9)
    report={'test':'offline delayed multi-reflection echo simulation',
        'far_end_source':'generated KIRA speech' if reference.is_file() else 'synthetic voiced signal',
        'echo_reduction_db':float(erle),'double_talk_output_to_user_rms_ratio':speech_ratio,
        'capture_processing_p95_ms':float(np.percentile(timings,95)*1000),
        'echo_suppressed':bool(erle>=12),'user_speech_retained':bool(speech_ratio>=.4),
        'under_20ms_frame_budget':bool(np.percentile(timings,95)<.02)}
    print(json.dumps(report,indent=2),flush=True)
    if not all(report[k] for k in ['echo_suppressed','user_speech_retained','under_20ms_frame_budget']):
        raise SystemExit(2)

if __name__=='__main__': main()
