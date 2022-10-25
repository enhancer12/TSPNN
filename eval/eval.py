import torch
import numpy as np
import librosa
import onnxruntime as ort
import argparse
import glob
import os
import math
from pypesq import pesq

import pandas as pd

from tqdm import tqdm

# SCENARIOS = [
#     'doubletalk-with-movement',
#     'doubletalk',
#     'farend-singletalk-with-movement',
#     'farend-singletalk',
#     'nearend-singletalk'
# ]
SCENARIOS = [
    'doubletalk',
    'farend',
    'earphone',
    'nearend'
]
SAMPLE_RATE = 16000

def compute_erle(echo_data, error_data):  # for far single,echo:mic,error:aec_out
    if len(echo_data) == 0 or len(error_data) == 0:
        return 0
    echo_ms = np.mean(np.power(echo_data, 2))
    error_ms = np.mean(np.power(error_data, 2))
    erle = 50
    if error_ms != 0:
        erle = 10 * math.log10(echo_ms / error_ms)
    return erle

class AECMOSEstimator():
    DFT_SIZE = 512
    HOP_FRACTION = 0.5

    def __init__(self, model_path):
        self.model_path = model_path
        self.max_len = 20
        self.sampling_rate = 16000
        if 'Run1644323924_Stage-0' in self.model_path:
            self.transform = self._mel_transform
        elif 'Run_1657188842_Stage_0' in self.model_path:
            self.transform = self._mel_transform
        else:
            ValueError, "Not a supported model."

    def _mel_transform(self, sample, sr):
        mel_spec = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=512 + 1, hop_length=256, n_mels=160)
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def run(self, talk_type, lpb_sig, mic_sig, enh_sig):
        assert len(lpb_sig) == len(mic_sig) == len(enh_sig)

        # cut segments if too long
        seg_nb_samples = self.max_len * self.sampling_rate
        if len(lpb_sig) >= seg_nb_samples:
            lpb_sig = lpb_sig[: seg_nb_samples]
            mic_sig = mic_sig[: seg_nb_samples]
            enh_sig = enh_sig[: seg_nb_samples]

        # feature transform
        lpb_sig = self.transform(lpb_sig, self.sampling_rate)
        mic_sig = self.transform(mic_sig, self.sampling_rate)
        enh_sig = self.transform(enh_sig, self.sampling_rate)

        # scenario marker
        assert talk_type in ['nst', 'st', 'dt']

        if talk_type == 'nst':
            ne_st = 1
            fe_st = 0
        elif talk_type == 'st':
            ne_st = 0
            fe_st = 1
        else:
            ne_st = 0
            fe_st = 0

        mic_sig = np.concatenate(
            (mic_sig, np.ones((20, mic_sig.shape[1])) * (1 - fe_st), np.zeros((20, mic_sig.shape[1]))), axis=0)
        lpb_sig = np.concatenate(
            (lpb_sig, np.ones((20, lpb_sig.shape[1])) * (1 - ne_st), np.zeros((20, lpb_sig.shape[1]))), axis=0)
        enh_sig = np.concatenate((enh_sig, np.ones((20, enh_sig.shape[1])), np.zeros((20, enh_sig.shape[1]))), axis=0)

        # stack
        feats = np.stack((lpb_sig, mic_sig, enh_sig)).astype(np.float32)
        feats = np.expand_dims(feats, axis=0)

        # model_input = feats
        ort_session = ort.InferenceSession(self.model_path)
        input_name = ort_session.get_inputs()[0].name

        # GRU hidden layer shape is in h0
        with torch.no_grad():
            h0 = torch.zeros((4, 1, 64), dtype=torch.float32).detach().numpy()
        result = ort_session.run([], {input_name: feats, 'h0': h0})
        result = result[0]

        echo_mos = float(result[0])
        deg_mos = float(result[1])
        return {'echo_mos': echo_mos, 'deg_mos': deg_mos}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Path to directory containing enhanced nearend speech')
    parser.add_argument('-d', '--dataset', required=True,
                        help='Path to directory containing loopback and mic files, needs to be same relative level as "input"')
    parser.add_argument(
        '--score_file', help='Path to file for saving scores, optional')
    parser.add_argument('--model_path', type=str, help="Specify the path to the onnx model provided")
    parser.add_argument('--offset_t', type=float, default=0.0, help="Specify the offset of error output in ms")
    parser.add_argument('--worker', type=int, default=40, help="Specify the number of workers for eval")

    return parser.parse_args()

def step(input_dir, dataset_dir, clip_path, aec_moser, offset_t):
    lpb_path, mic_path = get_lpb_mic_paths(
        input_dir, dataset_dir, clip_path)

    lpb_sig, mic_sig, enh_sig, tar_sig = read_and_process_audio_files(
        lpb_path, mic_path, clip_path, offset_t)

    talk_type = get_clip_scenario(os.path.basename(clip_path))

    if 'double' in talk_type:
        talk_type = 'dt'
    elif 'farend' in talk_type:
        talk_type = 'st'
    elif ('nearend' in talk_type) or ('earphone' in talk_type):
        talk_type = 'nst'
    else:
        assert 0, f"Unknown type: {talk_type}"

    clip_scores = aec_moser.run(talk_type, 
                            lpb_sig, mic_sig, enh_sig)

    erle = -1
    if talk_type == 'st':
        erle = compute_erle(mic_sig, enh_sig)

    pesq_tmp = -1
    if talk_type != 'st' and tar_sig is not None:
        pesq_tmp = pesq(tar_sig, enh_sig, SAMPLE_RATE)

    score = {'file': os.path.basename(clip_path), 'talk_type': talk_type, 'erle': erle, 'pesq': pesq_tmp}
    score.update(clip_scores)

    return score

def main(input_dir, dataset_dir, score_file, aec_moser, offset_t, worker=40):
    clips = glob.glob(os.path.join(input_dir, '**', '*.wav'), recursive=True)

    import multiprocessing as mp

    pool = mp.Pool(worker)
    
    pbar = tqdm(total=len(clips))
    pbar.set_description("Compute metrics")
    pbar_update = lambda *args: pbar.update()

    results = [pool.apply_async(step, args=(input_dir, dataset_dir, _, aec_moser, offset_t), callback=pbar_update) \
                for idx, _ in enumerate(clips)]
    scores = [p.get() for p in results]
    pool.close()
    pool.join()

    direct_out = [0] * 5
    print('\n')
    for type_ in ['st', 'nst', 'dt']:
        tmp_scores =  [_ for _ in scores if _['talk_type']==type_]
        tmp_df = pd.DataFrame(tmp_scores)

        if type_ == 'st':
            direct_out[0] = tmp_df.erle.mean()
            direct_out[1] = tmp_df.echo_mos.mean()
        elif type_ == 'nst':
            direct_out[4] = tmp_df.deg_mos.mean()
        else:
            direct_out[2] = tmp_df.echo_mos.mean()
            direct_out[3] = tmp_df.deg_mos.mean()

        print(
        f'{type_}: Mean echo MOS is {tmp_df.echo_mos.mean():.4}, other degradation MOS is {tmp_df.deg_mos.mean():.4}, erle: {tmp_df.erle.mean():.4}, pesq: {tmp_df.pesq.mean():.4}')

    # print(f"{direct_out[0]:.4} {direct_out[1]:.4} {direct_out[2]:.4} {direct_out[3]:.4} {direct_out[4]:.4} {np.mean(direct_out[1:]):.4}")
    print(f"AECMOS(avg): {np.mean(direct_out[1:]):.4}")

    scores_df = pd.DataFrame(scores)

    if score_file:
        scores_df.to_csv(score_file, index=False)


def get_lpb_mic_paths(input_dir, dataset_dir, clip_path):
    sub_dir, basename = os.path.split(clip_path)
    if 'est.wav' in basename:
        basename = basename.replace('est', 'mic')
    lpb_path = os.path.join(dataset_dir, os.path.split(sub_dir)[-1], basename.replace('mic', 'lpb'))
    mic_path = os.path.join(dataset_dir, os.path.split(sub_dir)[-1], basename)

    #print(f"\nlpb: {lpb_path}\nmic: {mic_path}\nenh: {clip_path}\n")
    return lpb_path, mic_path


def get_clip_hash(clip_file_name):
    # In case a path is passed
    input_parts = os.path.split(clip_file_name)
    return input_parts[-1][:22]


def get_clip_scenario(clip_file_name):
    for s in SCENARIOS:
        if s in clip_file_name:
            return s

    assert 0, f"Unknown type: {clip_file_name}"

    return ""


def read_and_process_audio_files(lpb_path, mic_path, clip_path, offset_t):
    lpb_sig, _ = librosa.load(lpb_path, sr=SAMPLE_RATE)
    mic_sig, _ = librosa.load(mic_path, sr=SAMPLE_RATE)
    enh_sig, _ = librosa.load(clip_path, sr=SAMPLE_RATE)

    clean_path = mic_path.replace('_mic.wav', '_clean.wav')
    tar_sig = None
    if os.path.exists(clean_path):
        tar_sig, _ = librosa.load(clean_path, sr=SAMPLE_RATE)

    # Make the clips the same length
    min_len = np.min([len(lpb_sig), len(mic_sig), len(enh_sig)])
    lpb_sig = lpb_sig[:min_len]
    mic_sig = mic_sig[:min_len]
    enh_sig = enh_sig[:min_len]

    if tar_sig is not None:
        tar_sig = tar_sig[:min_len]

    #print(clip_path)
    if True: #is_blind_testset(clip_path):
        #assert 0
        lpb_sig, mic_sig, enh_sig, tar_sig = process_blind_testset(
            lpb_sig, mic_sig, enh_sig, tar_sig, clip_path)

    # set offset
    
    delay = int(SAMPLE_RATE / 1000 * offset_t)
    if delay > 0:
        enh_sig = enh_sig[delay:]

        lpb_sig = lpb_sig[:-delay]
        mic_sig = mic_sig[:-delay]

        if tar_sig is not None:
            tar_sig = tar_sig[:-delay]
    elif delay < 0:
        delay = abs(delay)
        enh_sig = enh_sig[:-delay]

        lpb_sig = lpb_sig[delay:]
        mic_sig = mic_sig[delay:]
        if tar_sig is not None:
            tar_sig = tar_sig[delay:]
    return lpb_sig, mic_sig, enh_sig, tar_sig


def is_blind_testset(clip_path):
    # This function can be used with your own custom heuristic
    # Here we assume the dataset name is somewhere on the full clip path
    return 'blind' in clip_path


def process_blind_testset(lpb_sig, mic_sig, enh_sig, tar_sig, clip_path):
    clip_name = os.path.basename(clip_path)
    clip_scenario = get_clip_scenario(clip_name)
    if clip_scenario in ['doubletalk-with-movement', 'doubletalk']:
        silence_duration = 15 * SAMPLE_RATE  # in seconds
        rating_dt_length = int((len(enh_sig) - silence_duration) / 2)

        if rating_dt_length > 0:
            lpb_sig = lpb_sig[-rating_dt_length:]
            mic_sig = mic_sig[-rating_dt_length:]
            enh_sig = enh_sig[-rating_dt_length:]

            if tar_sig is not None:
                tar_sig = tar_sig[-rating_dt_length:]

    elif clip_scenario in ['farend-singletalk-with-movement', 'farend-singletalk', 'farend']:
        rating_fest_length = int(len(enh_sig) / 2)

        lpb_sig = lpb_sig[-rating_fest_length:]
        mic_sig = mic_sig[-rating_fest_length:]
        enh_sig = enh_sig[-rating_fest_length:]

        if tar_sig is not None:
            tar_sig = tar_sig[-rating_fest_length:]

    elif clip_scenario in ['nearend-singletalk', 'nearend', 'earphone']:
        pass
    else:
        raise Exception()
    
    return lpb_sig, mic_sig, enh_sig, tar_sig


if __name__ == '__main__':
    args = parse_args()

    input_dir = args.input
    dataset_dir = args.dataset
    score_file = args.score_file
    model_path = args.model_path
    offset_t = args.offset_t
    worker = args.worker

    import pprint
    pprint.pprint(args)

    aecmos = AECMOSEstimator(model_path)

    main(input_dir, dataset_dir, score_file, aecmos, offset_t, worker)
