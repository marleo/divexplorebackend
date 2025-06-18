import torch
import open_clip
from PIL import Image
import numpy as np
import os
import sys
import glob
import csv

imsuffix = 'png'
if len(sys.argv) < 3:
    print("please specify <folder with images> and <result-filename-no-suffix> [<image-suffix]")
    exit(1)

if len(sys.argv) > 3:
    imsuffix = sys.argv[3]

rootdir = sys.argv[1]

modelname = 'ViT-H-14-378-quickgelu' #'ViT-H-14' #'ViT-L-14' #'ViT-bigG-14' #'ViT-g-14' #'ViT-L-14' #'ViT-B-32' #'ViT-H-14'
modelweights = 'dfn5b' #'laion2b_s32b_b79k' #'laion400m_e32' #'laion2b_s39b_b160k' #'laion2b_s12b_b42k' #'laion2b_s32b_b82k' #'laion2b_s34b_b79k' #'laion2b_s32b_b79k'
modelname = 'ViT-H-14' #'ViT-L-14' #'ViT-bigG-14' #'ViT-g-14' #'ViT-L-14' #'ViT-B-32' #'ViT-H-14'
modelweights = 'laion2b_s32b_b79k' #'laion400m_e32' #'laion2b_s39b_b160k' #'laion2b_s12b_b42k' #'laion2b_s32b_b82k' #'laion2b_s34b_b79k' #'laion2b_s32b_b79k'
#modelname = 'ViT-B-16' #'ViT-L-14' #'ViT-bigG-14' #'ViT-g-14' #'ViT-L-14' #'ViT-B-32' #'ViT-H-14'
#modelweights = 'laion2b_s34b_b88k' #'laion400m_e32' #'laion2b_s39b_b160k' #'laion2b_s12b_b42k' #'laion2b_s32b_b82k' #'laion2b_s34b_b79k' #'laion2b_s32b_b79k'

device = "cuda" if torch.cuda.is_available() else "cpu"
#model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
model, _, preprocess = open_clip.create_model_and_transforms(modelname, pretrained=modelweights, device=device)

print(f'model loaded with {device}!')

#for root, subFolders, files in os.walk(rootdir):
#    #root is currently walked dir
#    #print(f'root: {root}')
#
#    #subFolders contains items in root of type dir
#    #for subdir in subFolders:
#    #    print(f'{subdir}')
#    
#    for filename in files:
#        print(f'{filename}')
 
csvfile = open(f'openclip-{sys.argv[2]}-{modelname}_{modelweights}.csv','w')
writer = csv.writer(csvfile, delimiter=',')

targetfiledir = rootdir + f'**/*.{imsuffix}'
print(targetfiledir)

for filename in glob.iglob(targetfiledir, recursive=True):
    basename = os.path.basename(filename)
    relpath = os.path.relpath(filename, rootdir)
    print(filename)
    #print(basename)


    image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
    #text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        #text_features = model.encode_text(text)
  
        image_features = image_features.cpu()
        #print(image_features[0].tolist())
        mylist = image_features[0].tolist()
        #mylist.insert(0, basename)
        mylist.insert(0, relpath)
        writer.writerow(mylist)
        #print(str(len(image_features[0])) + " features")
        #featuresfilename = os.path.basename(filename).replace(".png", "-clip.txt") 
        #np.savetxt(os.path.join("clipfeatures/", featuresfilename), image_features[0].numpy())

        print(relpath)
        #logits_per_image, logits_per_text = model(image, text)
        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    #print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
csvfile.close()
