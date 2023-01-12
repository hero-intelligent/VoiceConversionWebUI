import numpy as np,ffmpeg,os,traceback
from slicer import Slicer
slicer = Slicer(
    sr=40000,
    db_threshold=-32,
    min_length=800,
    win_l=400,
    win_s=20,
    max_silence_kept=150
)




def p0_load_audio(file, sr):#str-ing
    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def p1_trim_audio(slicer,audio):return slicer.slice(audio)

def p2_avg_cut(audio,sr,per=3.7,overlap=0.3,tail=4):
    i = 0
    audios=[]
    while (1):
        start = int(sr * (per - overlap) * i)
        i += 1
        if (len(audio[start:]) > tail * sr):
            audios.append(audio[start:start + int(per * sr)])
        else:
            audios.append(audio[start:])
            break
    return audios

def p2b_get_vol(audio):return np.square(audio).mean()

def p3_norm(audio,alpha=0.8,maxx=0.95):return audio / np.abs(audio).max() * (maxx * alpha) + (1-alpha) * audio

def pipeline(inp_root,sr1=40000,sr2=16000,if_trim=True,if_avg_cut=True,if_norm=True,save_root1=None,save_root2=None):
    if(save_root1==None and save_root2==None):return "No save root."
    name2vol={}
    infos=[]
    names=[]
    for name in os.listdir(inp_root):
        try:
            inp_path=os.path.join(inp_root,name)
            audio=p0_load_audio(inp_path)
        except:
            infos.append("%s\t%s"%(name,traceback.format_exc()))
            continue
        if(if_trim==True):res1s=p1_trim_audio(audio)
        else:res1s=[audio]
        for i0,res1 in res1s:
            if(if_avg_cut==True):res2=p2_avg_cut(res1)
            else:res2=[res1]


