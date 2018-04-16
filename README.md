# impression_personality

## Cropped Photos

Clone the repo and make sure that the crop.py script and haar cascade xml files are in the same directory as the folder containing all the data. 

Currently the script expects the images to be in VC/pics/vc and VC/pics/e. These directories were renamed as they were easier for me to specify their names since I wanted to test without escaping the space charaacters. In order to make this script work, just rename the appropriate folders or change the appropriate parts of the e_path and vc_path variables.

The script should then create two items per vc[num] and e[num]
1) a .txt file containing a list of all the photos that should be categorized as invalid, along with their category 
2) a folder containing cropped images of the photos that it determined was valid
