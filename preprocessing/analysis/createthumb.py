import cv2 as cv
import sys
import os

OUTPATH = "../../thumbs"
#print( len(sys.argv) )

if len(sys.argv) < 2:
    print("usage: createthumb.py <keyframe>")
    exit(0)

fkeyframe = sys.argv[1]
keyframe = cv.imread(fkeyframe)
fthumb = fkeyframe.replace("../../keyframes",OUTPATH).replace(".png",".jpg")

comps = fkeyframe.split(os.sep)
videoid = comps[3]

viddir = os.path.join(OUTPATH,videoid)
if not os.path.exists(viddir):
    os.mkdir(viddir)

if not os.path.exists(fthumb):
    print(fthumb)
    thumb = cv.resize(src=keyframe, dsize=(160, 90))
    cv.imwrite(fthumb,thumb)
