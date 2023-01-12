import torch, pdb, os,traceback,sys,warnings,shutil
now_dir=os.getcwd()
sys.path.append(now_dir)
tmp=os.path.join(now_dir,"TEMP")
shutil.rmtree(tmp,ignore_errors=True)
os.makedirs(tmp,exist_ok=True)
os.environ["TEMP"]=tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from infer_pack.models import SynthesizerTrnMs256NSF as SynthesizerTrn256
from scipy.io import wavfile
from fairseq import checkpoint_utils
import gradio as gr
import librosa
import logging
from vc_infer_pipeline import VC
import soundfile as sf
from config import is_half,device,is_half
from infer_uvr5 import _audio_pre_
logging.getLogger('numba').setLevel(logging.WARNING)

models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"],suffix="",)
hubert_model = models[0]
hubert_model = hubert_model.to(device)
if(is_half):hubert_model = hubert_model.half()
else:hubert_model = hubert_model.float()
hubert_model.eval()


weight_root="weights"
weight_uvr5_root="uvr5_weights"
names=[]
for name in os.listdir(weight_root):names.append(name.replace(".pt",""))
uvr5_names=[]
for name in os.listdir(weight_uvr5_root):uvr5_names.append(name.replace(".pth",""))

def get_vc(sid):
    person = "%s/%s.pt" % (weight_root, sid)
    cpt = torch.load(person, map_location="cpu")
    dv = cpt["dv"]
    tgt_sr = cpt["config"][-1]
    net_g = SynthesizerTrn256(*cpt["config"], is_half=is_half)
    net_g.load_state_dict(cpt["weight"], strict=True)
    net_g.eval().to(device)
    if (is_half):net_g = net_g.half()
    else:net_g = net_g.float()
    vc = VC(tgt_sr, device, is_half)
    return dv,tgt_sr,net_g,vc

def vc_single(sid,input_audio,f0_up_key,f0_file):
    if input_audio is None:return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        if(type(input_audio)==str):
            print("processing %s" % input_audio)
            audio, sampling_rate = sf.read(input_audio)
        else:
            sampling_rate, audio = input_audio
            audio = audio.astype("float32") / 32768
        if(type(sid)==str):dv, tgt_sr, net_g, vc=get_vc(sid)
        else:dv,tgt_sr,net_g,vc=sid
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        times = [0, 0, 0]
        audio_opt=vc.pipeline(hubert_model,net_g,dv,audio,times,f0_up_key,f0_file=f0_file)
        print(times)
        return "Success", (tgt_sr, audio_opt)
    except:
        info=traceback.format_exc()
        print(info)
        return info,(None,None)
    finally:
        print("clean_empty_cache")
        del net_g,dv,vc
        torch.cuda.empty_cache()

def vc_multi(sid,dir_path,opt_root,paths,f0_up_key):
    try:
        dir_path=dir_path.strip(" ")#防止小白拷路径头尾带了空格
        opt_root=opt_root.strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        dv, tgt_sr, net_g, vc = get_vc(sid)
        try:
            if(dir_path!=""):paths=[os.path.join(dir_path,name)for name in os.listdir(dir_path)]
            else:paths=[path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos=[]
        for path in paths:
            info,opt=vc_single([dv,tgt_sr,net_g,vc],path,f0_up_key,f0_file=None)
            if(info=="Success"):
                try:
                    tgt_sr,audio_opt=opt
                    wavfile.write("%s/%s" % (opt_root, os.path.basename(path)), tgt_sr, audio_opt)
                except:
                    info=traceback.format_exc()
            infos.append("%s->%s"%(os.path.basename(path),info))
        return "\n".join(infos)
    except:
        return traceback.format_exc()
    finally:
        print("clean_empty_cache")
        del net_g,dv,vc
        torch.cuda.empty_cache()

def uvr(model_name,inp_root,save_root_vocal,save_root_ins):
    infos = []
    try:
        inp_root = inp_root.strip(" ")# 防止小白拷路径头尾带了空格
        save_root_vocal = save_root_vocal.strip(" ")
        save_root_ins = save_root_ins.strip(" ")
        pre_fun = _audio_pre_(model_path=os.path.join(weight_uvr5_root,model_name+".pth"), device=device, is_half=is_half)
        for name in os.listdir(inp_root):
            inp_path=os.path.join(inp_root,name)
            try:
                pre_fun._path_audio_(inp_path , save_root_ins,save_root_vocal)
                infos.append("%s->Success"%(os.path.basename(inp_path)))
            except:
                infos.append("%s->%s" % (os.path.basename(inp_path),traceback.format_exc()))
    except:
        infos.append(traceback.format_exc())
    finally:
        try:
            del pre_fun.model
            del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        torch.cuda.empty_cache()
    return "\n".join(infos)

with gr.Blocks() as app:
    with gr.Tabs():
        with gr.TabItem("推理"):
            with gr.Group():
                gr.Markdown(value="""
                    使用软件者、传播软件导出的声音者自负全责。如不认可该条款，则不能使用/引用软件包内所有代码和文件。<br>
                    目前仅开放白菜音色，后续将扩展为本地训练推理工具，用户可训练自己的音色进行社区共享。<br>
                    男转女推荐+12key，女转男推荐-12key，如果音域爆炸导致音色失真也可以自己调整到合适音域
                    """)
                with gr.Row():
                    with gr.Column():
                        sid0 = gr.Dropdown(label="音色", choices=names)
                        vc_transform0 = gr.Number(label="变调（整数，半音数量，升八度12降八度-12）", value=12)
                        f0_file = gr.File(label="F0曲线文件，可选，一行一个音高，代替默认F0及升降调")
                    input_audio0 = gr.Audio(label="上传音频")
                    but0=gr.Button("转换", variant="primary")
                    with gr.Column():
                        vc_output1 = gr.Textbox(label="输出信息")
                        vc_output2 = gr.Audio(label="输出音频")
                    but0.click(vc_single, [sid0, input_audio0, vc_transform0,f0_file], [vc_output1, vc_output2])
            with gr.Group():
                gr.Markdown(value="""
                    批量转换，上传多个音频文件，在指定文件夹（默认opt）下输出转换的音频。<br>
                    合格的文件夹路径格式举例：E:\codes\py39\\vits_vc_gpu\白鹭霜华测试样例（去文件管理器地址栏拷就行了）
                    """)
                with gr.Row():
                    with gr.Column():
                        sid1 = gr.Dropdown(label="音色", choices=names)
                        vc_transform1 = gr.Number(label="变调（整数，半音数量，升八度12降八度-12）", value=12)
                        opt_input = gr.Textbox(label="指定输出文件夹",value="opt")
                    with gr.Column():
                        dir_input = gr.Textbox(label="输入待处理音频文件夹路径")
                        inputs = gr.File(file_count="multiple", label="也可批量输入音频文件，二选一，优先读文件夹")
                    but1=gr.Button("转换", variant="primary")
                    vc_output3 = gr.Textbox(label="输出信息")
                    but1.click(vc_multi, [sid1, dir_input,opt_input,inputs, vc_transform1], [vc_output3])

        with gr.TabItem("数据处理"):
            with gr.Group():
                gr.Markdown(value="""
                    人声伴奏分离批量处理，使用UVR5模型。<br>
                    不带和声用HP2，带和声且提取的人声不需要和声用HP5<br>
                    合格的文件夹路径格式举例：E:\codes\py39\\vits_vc_gpu\白鹭霜华测试样例（去文件管理器地址栏拷就行了）
                    """)
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(label="输入待处理音频文件夹路径")
                        wav_inputs = gr.File(file_count="multiple", label="也可批量输入音频文件，二选一，优先读文件夹")
                    with gr.Column():
                        model_choose = gr.Dropdown(label="模型", choices=uvr5_names)
                        opt_vocal_root = gr.Textbox(label="指定输出人声文件夹",value="opt")
                        opt_ins_root = gr.Textbox(label="指定输出乐器文件夹",value="opt")
                    but2=gr.Button("转换", variant="primary")
                    vc_output4 = gr.Textbox(label="输出信息")
                    but2.click(uvr, [model_choose, dir_wav_input,opt_vocal_root,opt_ins_root], [vc_output4])
        with gr.TabItem("训练-待开放"):pass

    # app.launch(server_name="0.0.0.0",server_port=7860)
    app.launch(server_name="127.0.0.1",server_port=7860)