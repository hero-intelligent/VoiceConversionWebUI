import numpy as np,parselmouth,torch,pdb
from time import time as ttime
import torch.nn.functional as F
from config import x_pad,x_query,x_center,x_max
from sklearn.cluster import KMeans

def resize2d(x, target_len,is1):
    minn=1 if is1==True else 0
    ss = np.array(x).astype("float32")
    ss[ss <=minn] = np.nan
    target = np.interp(np.arange(0, len(ss) * target_len, len(ss)) / target_len, np.arange(0, len(ss)), ss)
    res = np.nan_to_num(target)
    return res

class VC(object):
    def __init__(self,tgt_sr,device,is_half):
        self.sr=16000#hubert输入采样率
        self.window=160#每帧点数
        self.t_pad=self.sr*x_pad#每条前后pad时间
        self.t_pad_tgt=tgt_sr*x_pad
        self.t_pad2=self.t_pad*2
        self.t_query=self.sr*x_query#查询切点前后查询时间
        self.t_center=self.sr*x_center#查询切点位置
        self.t_max=self.sr*x_max#免查询时长阈值
        self.device=device
        self.is_half=is_half

    def get_f0(self,x, p_len,f0_up_key=0,inp_f0=None):
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = parselmouth.Sound(x, self.sr).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0=self.sr//self.window#每秒f0点数
        if (inp_f0 is not None):
            delta_t=np.round((inp_f0[:,0].max()-inp_f0[:,0].min())*tf0+1).astype("int16")
            replace_f0=np.interp(list(range(delta_t)), inp_f0[:, 0]*100, inp_f0[:, 1])
            shape=f0[x_pad*tf0:x_pad*tf0+len(replace_f0)].shape[0]
            f0[x_pad*tf0:x_pad*tf0+len(replace_f0)]=replace_f0[:shape]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)
        return f0_coarse, f0bak#1-0

    def vc(self,model,net_g,dv,audio0,pitch,pitchf,times):
        feats = torch.from_numpy(audio0)
        if(self.is_half==True):feats=feats.half()
        else:feats=feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": 9,  # layer 9
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats  = model.final_proj(logits[0])
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        t1 = ttime()
        p_len = audio0.shape[0]//self.window
        if(feats.shape[1]<p_len):
            p_len=feats.shape[1]
            pitch=pitch[:,:p_len]
            pitchf=pitchf[:,:p_len]
        p_len=torch.LongTensor([p_len]).to(self.device)
        with torch.no_grad():
            audio1 = (net_g.infer(feats, p_len, pitch, pitchf, dv)[0][0, 0] * 32768).data.cpu().float().numpy().astype(np.int16)
        del feats,p_len,padding_mask
        torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += (t1 - t0)
        times[2] += (t2 - t1)
        return audio1
    def vc_km(self,model,net_g,dv,audio0,pitch,pitchf,times):
        kmeans = KMeans(500)
        def get_cluster_result(x):
            """x: np.array [t, 256]"""
            return kmeans.predict(x)
        checkpoint = torch.load("lulu_contentvec_kmeans_500.pt")
        kmeans.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
        kmeans.__dict__["_n_threads"] = checkpoint["_n_threads"]
        kmeans.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"]
        feats = torch.from_numpy(audio0).float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.half().to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": 9,  # layer 9
        }
        torch.cuda.synchronize()
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0])
        feats = get_cluster_result(feats.cpu().numpy()[0].astype("float32"))
        feats = torch.from_numpy(feats).to(self.device)
        feats = F.interpolate(feats.half().unsqueeze(0).unsqueeze(0), scale_factor=2).long().squeeze(0)
        t1 = ttime()
        p_len = audio0.shape[0]//self.window
        if(feats.shape[1]<p_len):
            p_len=feats.shape[1]
            pitch=pitch[:,:p_len]
            pitchf=pitchf[:,:p_len]
        p_len=torch.LongTensor([p_len]).to(self.device)
        with torch.no_grad():
            audio1 = (net_g.infer(feats, p_len, pitch, pitchf, dv)[0][0, 0] * 32768).data.cpu().float().numpy().astype(np.int16)
        del feats,p_len,padding_mask
        torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += (t1 - t0)
        times[2] += (t2 - t1)
        return audio1

    def pipeline(self,model,net_g,dv,audio,times,f0_up_key,f0_file=None):
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode='reflect')
        opt_ts = []
        if(audio_pad.shape[0]>self.t_max):
            audio_sum = np.zeros_like(audio)
            for i in range(self.window): audio_sum += audio_pad[i:i - self.window]
            for t in range(self.t_center, audio.shape[0],self.t_center):opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query:t + self.t_query]) == np.abs(audio_sum[t - self.t_query:t + self.t_query]).min())[0][0])
        s = 0
        audio_opt=[]
        t=None
        t1=ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode='reflect')
        p_len=audio_pad.shape[0]//self.window
        inp_f0=None
        if(hasattr(f0_file,'name') ==True):
            try:
                with open(f0_file.name,"r")as f:
                    lines=f.read().strip("\n").split("\n")
                inp_f0=[]
                for line in lines:inp_f0.append([float(i)for i in line.split(",")])
                inp_f0=np.array(inp_f0,dtype="float32")
            except:
                traceback.print_exc()
        pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key,inp_f0)

        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        # if(inp_f0 is None):
        #     pitch = pitch[:p_len]
        #     pitchf = pitchf[:p_len]
        # else:
        #     pitch=resize2d(pitch,p_len,is1=True)
        #     pitchf=resize2d(pitchf,p_len,is1=False)
        pitch = torch.LongTensor(pitch).unsqueeze(0).to(self.device)
        pitchf = torch.FloatTensor(pitchf).unsqueeze(0).to(self.device)
        t2=ttime()
        times[1] += (t2 - t1)
        for t in opt_ts:
            t=t//self.window*self.window
            audio_opt.append(self.vc(model,net_g,dv,audio_pad[s:t+self.t_pad2+self.window],pitch[:,s//self.window:(t+self.t_pad2)//self.window],pitchf[:,s//self.window:(t+self.t_pad2)//self.window],times)[self.t_pad_tgt:-self.t_pad_tgt])
            s = t
        audio_opt.append(self.vc(model,net_g,dv,audio_pad[t:],pitch[:,t//self.window:]if t is not None else pitch,pitchf[:,t//self.window:]if t is not None else pitchf,times)[self.t_pad_tgt:-self.t_pad_tgt])
        audio_opt=np.concatenate(audio_opt)
        del pitch,pitchf
        return audio_opt
    def pipeline_km(self,model,net_g,dv,audio,times,f0_up_key,f0_file=None):
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode='reflect')
        opt_ts = []
        if(audio_pad.shape[0]>self.t_max):
            audio_sum = np.zeros_like(audio)
            for i in range(self.window): audio_sum += audio_pad[i:i - self.window]
            for t in range(self.t_center, audio.shape[0],self.t_center):opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query:t + self.t_query]) == np.abs(audio_sum[t - self.t_query:t + self.t_query]).min())[0][0])
        s = 0
        audio_opt=[]
        t=None
        t1=ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode='reflect')
        p_len=audio_pad.shape[0]//self.window
        inp_f0=None
        if(hasattr(f0_file,'name') ==True):
            try:
                with open(f0_file.name,"r")as f:
                    lines=f.read().strip("\n").split("\n")
                inp_f0=[]
                for line in lines:inp_f0.append([float(i)for i in line.split(",")])
                inp_f0=np.array(inp_f0,dtype="float32")
            except:
                traceback.print_exc()
        pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key,inp_f0)

        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        # if(inp_f0 is None):
        #     pitch = pitch[:p_len]
        #     pitchf = pitchf[:p_len]
        # else:
        #     pitch=resize2d(pitch,p_len,is1=True)
        #     pitchf=resize2d(pitchf,p_len,is1=False)
        pitch = torch.LongTensor(pitch).unsqueeze(0).to(self.device)
        pitchf = torch.FloatTensor(pitchf).unsqueeze(0).to(self.device)
        t2=ttime()
        times[1] += (t2 - t1)
        for t in opt_ts:
            t=t//self.window*self.window
            audio_opt.append(self.vc_km(model,net_g,dv,audio_pad[s:t+self.t_pad2+self.window],pitch[:,s//self.window:(t+self.t_pad2)//self.window],pitchf[:,s//self.window:(t+self.t_pad2)//self.window],times)[self.t_pad_tgt:-self.t_pad_tgt])
            s = t
        audio_opt.append(self.vc_km(model,net_g,dv,audio_pad[t:],pitch[:,t//self.window:]if t is not None else pitch,pitchf[:,t//self.window:]if t is not None else pitchf,times)[self.t_pad_tgt:-self.t_pad_tgt])
        audio_opt=np.concatenate(audio_opt)
        del pitch,pitchf
        return audio_opt
