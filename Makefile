make gif:
	ffmpeg -i output_video.mp4 -vf "setpts=1.5*PTS,scale=iw/2:-1" -r 15 output_1.gif
	ffmpeg -i output2.mp4 -vf "setpts=1.5*PTS,scale=iw/2:-1" -r 15 output_2.gif

