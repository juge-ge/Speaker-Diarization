import os
import ffmpeg
import torch
from datetime import timedelta
from funasr import AutoModel


def format_time(milliseconds):
    """格式化时间，超过1小时显示hh:mm:ss.mmm，否则显示mm:ss.mmm"""
    time_obj = timedelta(milliseconds=milliseconds)
    hours = time_obj.seconds // 3600
    minutes = (time_obj.seconds % 3600) // 60
    seconds = time_obj.seconds % 60
    milliseconds = milliseconds % 1000

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def audio_speaker_separation(input_audio_path, output_dir):
    """
    处理音频文件，生成转录文本并保存到txt文件中
    Args:
        input_audio_path (str): 输入音频路径
        output_dir (str): 输出文件夹路径
    Returns:
        str: 保存的转录文件路径
    """
    # 硬编码模型路径
    asr_model_path = "/home/data/data/iic/paraformer-zh"
    asr_model_revision = "v2.0.4"
    vad_model_path = "/home/data/jdssy_liy/Speaker Diarization/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    vad_model_revision = "v2.0.4"
    punc_model_path = "/home/data/data/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    punc_model_revision = "v2.0.4"
    spk_model_path = "/home/data/jdssy_liy/Speaker Diarization/speech_campplus_sv_zh-cn_16k-common"
    spk_model_revision = "v2.0.4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    try:
        model = AutoModel(
            model=asr_model_path,
            model_revision=asr_model_revision,
            vad_model=vad_model_path,
            vad_model_revision=vad_model_revision,
            punc_model=punc_model_path,
            punc_model_revision=punc_model_revision,
            spk_model=spk_model_path,
            spk_model_revision=spk_model_revision,
            ngpu=1 if torch.cuda.is_available() else 0,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {str(e)}")

    # 检查输入音频文件是否存在
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Audio file not found: {input_audio_path}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(input_audio_path) + "_transcription.txt")

    try:
        # 使用 ffmpeg 将音频转为16kHz单声道PCM
        audio_bytes, _ = (
            ffmpeg.input(input_audio_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 调用模型生成转录结果
        res = model.generate(input=audio_bytes, batch_size_s=300, is_final=True, sentence_timestamp=True)

        # 解析转录结果并生成输出文本
        transcription = "\n".join(
            f"{format_time(sentence['start'])} - {format_time(sentence['end'])} 说话人{sentence['spk']}: {sentence['text']}"
            for sentence in res[0]["sentence_info"]
        )

        # 将转录结果保存到文本文件中
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)

        print(f"Transcription saved to {output_file}")
        return output_file
    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")