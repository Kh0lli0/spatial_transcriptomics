# spatial_transcriptomics
# Base directory and filename
dir_base = '/athena/pascuallab/scratch/zuw4001/'
filename = 'CS_7588_JDM_286_TIF1g.TIF'
 
# Load the image
img = imread(dir_base + filename)
 
# If RGB or RGBA, convert to grayscale
if img.ndim == 3 and img.shape[-1] in [3,4]:
    img = rgb2gray(img)
 
# Normalize image by percentiles (5th and 95th)
img = normalize(img, pmin=5, pmax=95)
 
# Add channel dimension for model input (Y, X, 1)
img = np.expand_dims(img, axis=-1)
 
# Load pretrained StarDist model (expects 3 input channels)
model = StarDist2D.from_pretrained('2D_versatile_he')
 
# Replicate single channel to 3 channels if needed
if img.shape[-1] == 1 and model.config.n_channel_in == 3:
img = np.repeat(img, 3, axis=-1)
