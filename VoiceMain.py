import os
import librosa
import numpy as np
import Preprocess.VoicePreprocess as Pre
from algorithm.RNNGAN import RNNGAN
import evaluate

def train():
    Data_Dir = 'data/MIR1K'
    tensorboard_directory = 'algorithm/graphs/RNNGAN'
    log_directory = 'algorithm/log'
    train_log_filename = 'train_log.csv'
    train_log_filename1 = 'gsar.csv'
    clear_tensorboard = True
    model_directory = 'algorithm/model'
    model_filename = 'RNNGAN.ckpt'
    if not os.path.exists(tensorboard_directory):
        os.makedirs(tensorboard_directory)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    open(os.path.join(log_directory, train_log_filename), 'w').close()

    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    iterations=20000
    batch_size = 64
    sample_frames = 10
    num_rnn_layer = 3
    num_hidden_units = [512, 512, 512]
    gen_learning_rate = 0.0005
    dis_learning_rate = 0.00001



    stfts_mono_train, stfts_src1_train, stfts_src2_train,stfts_mono_valid, stfts_src1_valid, stfts_src2_valid=Pre.VoicePreprocessAll(Data_Dir=Data_Dir,mir1k_sr=mir1k_sr,n_fft = n_fft)
    model=RNNGAN(num_batch=batch_size,num_frames=sample_frames,n_fft=n_fft,num_rnn_layer=num_rnn_layer,num_hidden_units=num_hidden_units,tensorboard_directory=tensorboard_directory,clear_tensorboard=clear_tensorboard)
    for i in range(iterations):
        x_mixed,y1,y2=Pre.VoiceProprocessFrame(stfts_mono_train, stfts_src1_train, stfts_src2_train,
                                               sample_frames = sample_frames,batch_size = batch_size)
        gen_loss,dis_loss=model.train(x=x_mixed,y2=y2,gen_learning_rate=gen_learning_rate,
                                     dis_learning_rate=dis_learning_rate)

        if i%10==0:
            print('Step: %d Gen Loss: %f Dis Loss: %f'%(i,gen_loss,dis_loss))
        if i%200==0:
            print('==============================================')
            x_mixed, y1, y2 = Pre.VoiceProprocessFrame(stfts_mono_train, stfts_src1_train, stfts_src2_train,
                                                       sample_frames=sample_frames, batch_size=batch_size)
            y2_pred,validation_gen_loss,validation_dis_loss=model.validate(x=x_mixed,y2=y2)
            print('Step: %d Validation Gen Loss: %f Validation Dis Loss: %f' %(i, validation_gen_loss,validation_dis_loss))
            print('==============================================')
            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write('{},{},{},{},{}\n'.format(i, gen_loss, dis_loss,validation_gen_loss,validation_dis_loss))

        if i % 1000 == 0:
            model.save(directory = model_directory, filename = model_filename)

'''
        if i % 1000 == 0:
            # evaluate.generate_demo()
            # evaluate.tf.reset_default_graph()
            gnsdr,gsir,gsar = evaluate.Evaluate(batch_size=batch_size,model=model)
            with open(os.path.join(log_directory, train_log_filename1), 'a') as log_file:
                log_file.write('{},{},{},{}\n'.format(i, gnsdr, gsir,gsar)) 

'''




def use():
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 3
    num_hidden_units = [512, 512, 512]
    num_batch = 64
    num_frames = 10
    clear_tensorboard = False

    songs_dir='data/MIR1K'
    output_directory = 'algorithm/demo'
    tensorboard_directory = 'algorithm/graphs/RNNGAN'
    model_directory = 'algorithm/model'
    model_filename = 'RNNGAN.ckpt'
    model_filepath = os.path.join(model_directory, model_filename)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    song_filenames=list()
    for file in os.listdir(songs_dir):
        if file.endswith('.mp3'):
            song_filenames.append(os.path.join(songs_dir, file))
        if file.endswith('.wav'):
            song_filenames.append(os.path.join(songs_dir, file))
    wavs_mono = list()
    print(song_filenames)
    for filename in song_filenames:
        wav_mono, _ = librosa.load(filename, sr = mir1k_sr, mono = True)
        wavs_mono.append(wav_mono)
    stfts_mono = list()
    for wav_mono in wavs_mono:
        stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono.transpose())
    model=RNNGAN(num_batch=num_batch,num_frames=num_frames,n_fft=n_fft ,num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    model.load(filepath=model_filepath)
    for wav_filename, wav_mono, stft_mono in zip(song_filenames, wavs_mono, stfts_mono):

        wav_filename_dir = os.path.dirname(wav_filename)
        wav_filename_base = os.path.basename(wav_filename)
        wav_mono_filename = wav_filename_base.split('.')[0] + '_mono.wav'
        wav_src2_hat_filename = wav_filename_base.split('.')[0] + '_src2.wav'
        wav_mono_filepath = os.path.join(output_directory, wav_mono_filename)
        wav_src2_hat_filepath = os.path.join(output_directory, wav_src2_hat_filename)
        print('Processing %s ...' % wav_filename_base)
        stft_mono_magnitude, stft_mono_phase = Pre.sperate_magnitude_phase(data = stft_mono)
        stft_mono_magnitude = np.array([stft_mono_magnitude])
        frequency=int(stft_mono_magnitude.shape[1]/num_batch/num_frames)
        x_mix = Pre.VoiceFinalFrameIn(stft_mono_magnitude, 0)
        y2_pred_min = model.test(x=x_mix)
        y2_pred = np.reshape(y2_pred_min, (1, num_batch*num_frames, 513), order='C')
        for i in range (1,frequency):
            x_mix=Pre.VoiceFinalFrameIn(stft_mono_magnitude,i)
            y2_pred_min = model.test(x=x_mix)
            y2_pred_min=np.reshape(y2_pred_min,(1,num_batch*num_frames,513),order='C')
            y2_pred=np.append(y2_pred,y2_pred_min,axis=1)
        y2_stft_hat = Pre.combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase[0:num_batch*num_frames*frequency])
        y2_stft_hat = y2_stft_hat.transpose()
        y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)
        librosa.output.write_wav(wav_mono_filepath, wav_mono, mir1k_sr)
        librosa.output.write_wav(wav_src2_hat_filepath, y2_hat, mir1k_sr)

if __name__ == '__main__':
   train()
   # use()
