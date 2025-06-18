import cv2 as cv
import sys
import os

OUTPATH = "../../V3C/thumbsXL"
#print( len(sys.argv) )

if len(sys.argv) < 2:
    print("usage: createthumbXL.py <keyframe>")
    exit(0)

fkeyframe = sys.argv[1]
keyframe = cv.imread(fkeyframe)
fthumb = fkeyframe.replace("../../V3C/keyframes",OUTPATH).replace(".png",".jpg")

comps = fkeyframe.split(os.sep)
videoid = comps[len(comps)-2]

viddir = os.path.join(OUTPATH,videoid)
if not os.path.exists(viddir):
    os.mkdir(viddir)

if not os.path.exists(fthumb):
    print(f"{viddir} - {fthumb}")
    thumb = cv.resize(src=keyframe, dsize=(320, 180))
    cv.imwrite(fthumb,thumb)
