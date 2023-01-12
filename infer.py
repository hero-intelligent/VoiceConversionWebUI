import torch, pdb, os,sys,librosa,warnings,traceback
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
sys.path.append(os.getcwd())
from config import inp_root,opt_root,f0_up_key,person,is_half,device
os.makedirs(opt_root,exist_ok=True)
import soundfile as sf
from infer_pack.models import SynthesizerTrnMs256NSF as SynthesizerTrn256
from scipy.io import wavfile
from fairseq import checkpoint_utils
import scipy.signal as signal
from vc_infer_pipeline import VC

models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"],suffix="",)
model = models[0]
model = model.to(device)
if(is_half):model = model.half()
else:model = model.float()
model.eval()

cpt=torch.load(person,map_location="cpu")
dv=cpt["dv"]
tgt_sr=cpt["config"][-1]
net_g = SynthesizerTrn256(*cpt["config"],is_half=is_half)
net_g.load_state_dict(cpt["weight"],strict=True)
net_g.eval().to(device)
if(is_half):net_g = net_g.half()
else:net_g = net_g.float()

vc=VC(tgt_sr,device,is_half)

for name in os.listdir(inp_root):
    try:
        wav_path="%s\%s"%(inp_root,name)
        print("processing %s"%wav_path)
        audio, sampling_rate = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != vc.sr:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=vc.sr)

        times = [0, 0, 0]
        audio_opt=vc.pipeline(model,net_g,dv,audio,times,f0_up_key,f0_file=None)
        wavfile.write("%s/%s"%(opt_root,name), tgt_sr, audio_opt)
    except:
        traceback.print_exc()

print(times)
