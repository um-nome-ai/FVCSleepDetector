from moviepy import VideoFileClip
from time import sleep

full_video = "./tests/base.mp4"
current_duration = VideoFileClip(full_video).duration
divide_into_count = 10
single_duration = current_duration/divide_into_count
count = 0
current_video = f"./tests/t{count}.mp4"

while current_duration > single_duration:
    clip = VideoFileClip(full_video).subclipped(current_duration-single_duration, current_duration)
    current_duration -= single_duration
    current_video = f"./tests/t{count}.mp4"
    count += 1
    clip.write_videofile(current_video, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')

    print("-----------------###-----------------")