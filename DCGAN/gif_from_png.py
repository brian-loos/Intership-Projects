import imageio
import glob, os

gif_duration = 10
filenames = []
len_run = 469
for i in range(100):
    for j in [155,311,467]:
        filenames.append('sample_output_%3.1f.png'%(j+i*len_run))
num_images= 300
time_per_frame = gif_duration/num_images
with imageio.get_writer('movie.gif', mode='I',duration = time_per_frame) as writer:

    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
