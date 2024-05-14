import os
import time
import logging

from torch.utils.mobile_optimizer import optimize_for_mobile

from utils.prompt_making import make_prompt

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

current_folder = os.getcwd()
import os
import torch
from vocos import Vocos
import logging
import langid

langid.set_languages(['en', 'zh', 'ja'])

import pathlib
import platform

if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from utils.sentence_cutter import split_text_into_sentences

from macros import *

device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda", 0)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
url = 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'

checkpoints_dir = "./checkpoints/"

model_checkpoint_name = "vallex-checkpoint.pt"

model = None

codec = None

vocos = None

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()


def preload_models_cus():
    global model, codec, vocos
    if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)
    if not os.path.exists(os.path.join(checkpoints_dir, model_checkpoint_name)):
        import wget
        try:
            logging.info(
                "Downloading model from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt ...")
            # download from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt to ./checkpoints/vallex-checkpoint.pt
            wget.download("https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
                          out="./checkpoints/vallex-checkpoint.pt", bar=wget.bar_adaptive)
        except Exception as e:
            logging.info(e)
            raise Exception(
                "\n Model weights download failed, please go to 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'"
                "\n manually download model weights and put it to {} .".format(os.getcwd() + "\checkpoints"))
    # VALL-E
    model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)
    checkpoint = torch.load(os.path.join(checkpoints_dir, model_checkpoint_name), map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.eval()

    # Encodec
    codec = AudioTokenizer(device)
    print(f"preload_models codec {codec}")

    vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)


def cover_torch_script(text, prompt=None, language='auto', accent='no-accent'):
    preload_models_cus()
    global model, codec, vocos, text_tokenizer, text_collater
    text = text.replace("\n", "").strip(" ")
    # detect language
    if language == "auto":
        language = langid.classify(text)[0]
    lang_token = lang2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # load prompt
    if prompt is not None:
        prompt_path = prompt
        if not os.path.exists(prompt_path):
            prompt_path = "./presets/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            prompt_path = "./customs/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            raise ValueError(f"Cannot find prompt {prompt}")
        prompt_data = np.load(prompt_path)
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]

        # numpy to tensor
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    else:
        audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
        text_prompts = torch.zeros([1, 0]).type(torch.int32)
        lang_pr = lang if lang != 'mix' else 'en'

    enroll_x_lens = text_prompts.shape[-1]
    logging.info(f"synthesize text: {text}")
    print(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
    print(f"enroll_x_lens {enroll_x_lens.__class__}")
    #    text_tokens.to(device),
    #         text_tokens_lens.to(device),
    #         audio_prompts,
    #         enroll_x_lens=enroll_x_lens,
    #         top_k=-100,
    #         temperature=1,
    #         prompt_language=lang_pr,
    #         text_language=langs if accent == "no-accent" else lang,
    traced_model = torch.jit.trace_module(model, {
        "inference": (
        (text_tokens.to(device), text_tokens_lens.to(device), audio_prompts, torch.tensor(enroll_x_lens),))})
    # traced_script_module_optimized = optimize_for_mobile(traced_model)
    os.makedirs(f"{current_folder}/model/vall-e-x")
    output_path = f"{current_folder}/model/vall-e-x/model.ptl"

    print(f"output torch script path {output_path}")
    traced_model._save_for_lite_interpreter(output_path)
    pass


def make_prompt_cus(name, audio_path):
    start = time.time()
    print(f"name {name} audio_path {audio_path}")
    ### Use given transcript
    make_prompt(name=name, audio_prompt_path=audio_path)
    duration = time.time() - start
    print(f"make_prompt took {duration} ms")


def genrate_audio_cus(text, name, out_audio_path):
    # download and load all models
    preload_models()
    start = time.time()
    # tˈɜːn lˈɛft at ðə nˈɛkst ˌɪntəsˈɛkʃən and kəntˈɪnjuː stɹˈeɪt fɔː fˈaɪvhˈʌndɹɪd mˈiːtəz
    # təɹn lɛft æt ðə nɛkst ˌɪntəɹˈsɛkʃən ənd kənˈtɪnju stɹeɪt fəɹ faɪv ˈhənəɹd ˈmitəɹz.
    # tˈɜːn lˈɛft at ðə nˈɛkst ˌɪntəsˈɛkʃən and kəntˈɪnjuː stɹˈeɪt fɔː fˈaɪvhˈʌndɹɪd mˈiːtəz
    text_prompt = text
    audio_array = generate_audio(text_prompt, prompt=name)
    duration = time.time() - start
    print(f"generate_audio took {duration} ms")

    write_wav(out_audio_path, SAMPLE_RATE, audio_array)


if __name__ == '__main__':
    # torch.manual_seed(1)
    logging.basicConfig(level=logging.DEBUG)
    name = "obama"
    print(f"current_folder {current_folder}")
    make_prompt_cus(name, f"{current_folder}/resources/cut_obama_11.wav")
    genrate_audio_cus("Turn left at the next intersection and continue straight for 500 meters.", name,
                   f"{current_folder}/paimon_cloned.wav")
    cover_torch_script("Turn left at the next intersection and continue straight for 500 meters.", prompt=name)
