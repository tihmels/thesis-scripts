#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import speech_recognition as sr

TS15_AUDIO_FILE = "/Users/tihmels/TS/ts15/TV-20220610-2023-5600/audio/TV-20220610-2023-5600.webs.h264.wav"
TS100_AUDIO_SHOT_FILE = "/Users/tihmels/TS/ts100/TV-20220608-1832-2600/audio/shot_2.wav"
TS100_AUDIO_FILE = "/Users/tihmels/TS/ts100/TV-20220608-1832-2600/audio/TV-20220608-1832-2600.webs.h264.wav"

r = sr.Recognizer()

with sr.AudioFile(TS100_AUDIO_SHOT_FILE) as source:
    print("record audio ...", end=" ")
    audio_data = r.record(source)
    print("done")

try:
    print("extract text from audio ...")
    text = r.recognize_google(audio_data, language="de-DE", show_all=True)
    print(text)
    print("done")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
