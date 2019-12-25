# -*- coding: utf8 -*-
from pydub import AudioSegment
source_file_path='.mp3'
destin_path = '.wav'
sound = AudioSegment.from_mp3(source_file_path)
sound.export(destin_path,format ='wav')
