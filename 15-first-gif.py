# configure conda env first.
# install this. ran this cmd >>> pip install six
# and then, install moviepy version 1.0.3. ran this cmd >>> pip install moviepy==1.0.3
# do not run 'pip install moviepy' cmd. in won't works. 1.0.3 version is more stable.

from moviepy.editor import VideoFileClip

clip = VideoFileClip("/home/dongvin/Documents/git-test/input.mp4").subclip(0, 5)  # First 5 seconds
clip.write_gif("/home/dongvin/Documents/git-test/output.gif", fps=15)