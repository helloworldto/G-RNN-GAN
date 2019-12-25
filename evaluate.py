import librosa
import os
import numpy as np
import tensorflow as tf
from mir_eval.separation import bss_eval_sources
from algorithm.RNNGAN import RNNGAN
import Preprocess.VoicePreprocess as Pre

from Preprocess.VoicePreprocess import load_wavs, wavs_to_specs, prepare_data_full, sperate_magnitude_phase, \
    combine_magnitdue_phase, VoiceFinalFrameIn

def generate_demo():
    batch_size = 64
    sample_frames = 10
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 3
    num_hidden_units = [512, 512, 512]
    clear_tensorboard = False
    tensorboard_directory = 'algorithm/graphs/RNNGAN'
    model_directory = 'algorithm/model'
    model_filename = 'RNNGAN.ckpt'
    model_filepath = os.path.join(model_directory, model_filename)
    model_filepath = os.path.join(model_directory, model_filename)

    mir1k_dir = 'data/MIR1K'
    test_path = os.path.join(mir1k_dir, 'test.txt')

    with open(test_path, 'r') as text_file:
        content = text_file.readlines()
    wav_filenames = [file.strip() for file in content]

    #wav_filenames = ['small_test_data/yifen_4_10.wav', 'small_test_data/yifen_5_10.wav']
    output_directory = 'demo'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames, sr = mir1k_sr)

    stfts_mono, stfts_src1, stfts_src2 = wavs_to_specs(
        wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, n_fft = n_fft, hop_length = hop_length)

    stfts_mono_full, stfts_src1_full, stfts_src2_full = prepare_data_full(stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2)

    model=RNNGAN(num_batch=batch_size,num_frames=sample_frames,n_fft=n_fft,num_rnn_layer=num_rnn_layer,num_hidden_units=num_hidden_units,tensorboard_directory=tensorboard_directory,clear_tensorboard=clear_tensorboard)
    model.load(filepath = model_filepath)

    for wav_filename, wav_mono, stft_mono_full in zip(wav_filenames, wavs_mono, stfts_mono_full):

        wav_filename_dir = os.path.dirname(wav_filename)
        wav_filename_base = os.path.basename(wav_filename)
        wav_mono_filename = wav_filename_base.split('.')[0] + '_mono.wav'
        wav_src1_hat_filename = wav_filename_base.split('.')[0] + '_src1.wav'
        wav_src2_hat_filename = wav_filename_base.split('.')[0] + '_src2.wav'
        wav_mono_filepath = os.path.join(output_directory, wav_mono_filename)
        wav_src1_hat_filepath = os.path.join(output_directory, wav_src1_hat_filename)
        wav_src2_hat_filepath = os.path.join(output_directory, wav_src2_hat_filename)

        print('Processing %s ...' % wav_filename_base)

        stft_mono_magnitude, stft_mono_phase = sperate_magnitude_phase(data = stft_mono_full)
        stft_mono_magnitude = np.array([stft_mono_magnitude])

        frequency=int(stft_mono_magnitude.shape[1]/batch_size/sample_frames)
        x_mix = Pre.VoiceFinalFrameIn(stft_mono_magnitude, 0)
        y2_pred_min = model.test(x=x_mix)
        y2_pred = np.reshape(y2_pred_min, (1, batch_size*sample_frames, 513), order='C')
        for i in range (1,frequency):
            x_mix=Pre.VoiceFinalFrameIn(stft_mono_magnitude,i)
            y2_pred_min = model.test(x=x_mix)
            y2_pred_min=np.reshape(y2_pred_min,(1,batch_size*sample_frames,513),order='C')
            y2_pred=np.append(y2_pred,y2_pred_min,axis=1)
        y2_stft_hat = Pre.combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase[0:batch_size*sample_frames*frequency])
        y2_stft_hat = y2_stft_hat.transpose()
        y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)
        librosa.output.write_wav(wav_mono_filepath, wav_mono, mir1k_sr)
        librosa.output.write_wav(wav_src2_hat_filepath, y2_hat, mir1k_sr)
        # y2_pred = model.test(x = stft_mono_magnitude)
        # y2_stft_hat = combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase)
        #
        # y2_stft_hat = y2_stft_hat.transpose()
        #
        # y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)
        #
        # librosa.output.write_wav(wav_mono_filepath, wav_mono, mir1k_sr)
        # librosa.output.write_wav(wav_src2_hat_filepath, y2_hat, mir1k_sr)


def bss_eval_global(wavs_mono, wavs_src1, wavs_src2, wavs_src1_pred, wavs_src2_pred):

    assert len(wavs_mono) == len(wavs_src1) == len(wavs_src2) == len(wavs_src1_pred) == len(wavs_src2_pred)

    num_samples = len(wavs_mono)

    gnsdr = np.zeros(2)
    gsir = np.zeros(2)
    gsar = np.zeros(2)
    frames_total = 0

    for wav_mono, wav_src1, wav_src2, wav_src1_pred, wav_src2_pred in zip(wavs_mono, wavs_src1, wavs_src2, wavs_src1_pred, wavs_src2_pred):


        len_cropped = min(wav_src1_pred.shape[-1], wav_src2_pred.shape[-1])
        wav_src1_cropped = wav_src1[:len_cropped]
        wav_src2_cropped = wav_src2[:len_cropped]
        wav_mono_cropped = wav_mono[:len_cropped]
        wav_src1_pred = wav_src1_pred[:len_cropped]
        wav_src2_pred = wav_src2_pred[:len_cropped]

        sdr, sir, sar, _ = bss_eval_sources(reference_sources = np.asarray([wav_src1_cropped, wav_src2_cropped]), estimated_sources = np.asarray([wav_src1_cropped, wav_src2_pred]), compute_permutation = False)
        sdr_mono, _, _, _ = bss_eval_sources(reference_sources = np.asarray([wav_src1_cropped, wav_src2_cropped]), estimated_sources = np.asarray([wav_mono_cropped, wav_mono_cropped]), compute_permutation = False)

        nsdr = sdr - sdr_mono
        gnsdr += len_cropped * nsdr
        gsir += len_cropped * sir
        gsar += len_cropped * sar
        frames_total += len_cropped

    gnsdr = gnsdr / frames_total
    gsir = gsir / frames_total
    gsar = gsar / frames_total

    return gnsdr, gsir, gsar


def evaluate():

    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 3
     #change
    batch_size = 64
    sample_frames = 10
    num_hidden_units = [512, 512, 512]
    clear_tensorboard = False
    tensorboard_directory = 'algorithm/graphs/RNNGAN'
    model_directory = 'algorithm/model'
    model_filename = 'RNNGAN.ckpt'
    model_filepath = os.path.join(model_directory, model_filename)

    mir1k_dir = 'data/MIR1K'
    test_path = os.path.join(mir1k_dir, 'test.txt')

    with open(test_path, 'r') as text_file:
        content = text_file.readlines()
    wav_filenames = [file.strip() for file in content]

    #wav_filenames = ['small_test_data/yifen_4_10.wav', 'small_test_data/yifen_5_10.wav']
    output_directory = 'demo'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames, sr = mir1k_sr)

    stfts_mono, stfts_src1, stfts_src2 = wavs_to_specs(
    wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, n_fft = n_fft, hop_length = hop_length)

    stfts_mono_full, stfts_src1_full, stfts_src2_full = prepare_data_full(stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2)

    model=RNNGAN(num_batch=batch_size,num_frames=sample_frames,n_fft=n_fft,num_rnn_layer=num_rnn_layer,num_hidden_units=num_hidden_units,tensorboard_directory=tensorboard_directory,clear_tensorboard=clear_tensorboard)
    model.load(filepath = model_filepath)
    wavs_src2_pred = list()
    for wav_filename, wav_mono, stft_mono_full in zip(wav_filenames, wavs_mono, stfts_mono_full):
        stft_mono_magnitude, stft_mono_phase = sperate_magnitude_phase(data = stft_mono_full)
        stft_mono_magnitude = np.array([stft_mono_magnitude])
        frequency=int(stft_mono_magnitude.shape[1]/batch_size/sample_frames)
        x_mix = VoiceFinalFrameIn(stft_mono_magnitude, 0)
        y2_pred_min = model.test(x=x_mix)

        y2_pred = np.reshape(y2_pred_min, (1, batch_size*sample_frames, 513), order='C')
        for i in range (1,frequency):
            x_mix=VoiceFinalFrameIn(stft_mono_magnitude,i)
            y2_pred_min = model.test(x=x_mix)
            y2_pred_min=np.reshape(y2_pred_min,(1,batch_size*sample_frames,513),order='C')
            y2_pred=np.append(y2_pred,y2_pred_min,axis=1)
        y2_stft_hat = combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase[0:batch_size*sample_frames*frequency])
        y2_stft_hat = y2_stft_hat.transpose()
        y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)
        wavs_src2_pred.append(y2_hat)

    gnsdr, gsir, gsar = bss_eval_global(wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, wavs_src1_pred = wavs_src1, wavs_src2_pred = wavs_src2_pred)

    print('GNSDR:', gnsdr)
    print('GSIR:', gsir)
    print('GSAR:', gsar)



if __name__ == '__main__':

    generate_demo()
    tf.reset_default_graph()
    evaluate()
