import glob
import os
import cv2
import numpy as np

cwd = os.getcwd()

e_path = "%s/VC/pics/e"%cwd
vc_path = "%s/VC/pics/vc"%cwd

Errors = []
category1 = "1. Not a face / no face found"
category2 = "2. Face covered"
category3 = "3. Cannot open"
category6 = "6. Multiple faces"

def write_to_file(path, errors):
	file = open("%s/invalid_photos.txt"%path, "w")
	for error in errors:
		file.write("%s\t%s\n"%(error))

	file.close()

def analyze_photos(path):
	
	cropped_images_path = "%s/cropped_photos"%path
	cropped_images_dir = os.makedirs(cropped_images_path, exist_ok=True)

	images = [glob.glob("%s/*"%path)][0]
	counter = 0

	for profile_pic in images:
		# print("Iteration %d with picture %s"% (counter, profile_pic))

		counter = counter + 1
		img = cv2.imread(profile_pic)
		if img is None: 
			Errors.append( (profile_pic, category3) )
			continue
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		# 	
		#	-- Skip Invalid Image: No faces or more than 1 face found
		# 	
		if len(faces) < 1:
			Errors.append( (profile_pic, category1) )
			continue

		
		for (x, y, w, h) in faces:
			#cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

			#cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

			glasses = glasses_cascade.detectMultiScale(roi_gray)

			#
			#	-- Skip Invalid Image: Person is wearing glasses
			#
			#if len(glasses) > 1: 
			#	continue;

			#for (gx, gy, gw, gh) in glasses:
			#	cv2.rectangle(roi_color, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)

		lenX = ((x+w) - x) // 2
		lenY = ((y+h) - y) // 2

		midX = (x + (x+w)) // 2
		midY = (y + (y+h)) // 2

		distX = lenX + (lenX // 2) 
		distY = lenY + (lenY // 2) 

		black_mask = np.zeros_like(img)
		white_mask = np.zeros_like(img)
		white_mask[:] = (255, 255, 255)
		
		white_mask = cv2.ellipse(white_mask, (midX, midY), (distX, distY), 0, 0, 360, (0,0,0), -1, 8)

		final_cropped = cv2.bitwise_or(img, white_mask)

		# cv2.imshow('img', final_cropped)
		# cv2.waitKey(0)

		filename = os.path.basename(profile_pic)
		cv2.imwrite("%s/%s"%(cropped_images_path,filename), final_cropped)

	write_to_file(path, Errors)
	Errors.clear()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
glasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

e_folders = [dir[0] for dir in os.walk(e_path)]
vc_folders = [dir[0] for dir in os.walk(vc_path)]

'''
for e_dir in e_folders:
	analyze_photos(e_dir)
'''

for v_dir in vc_folders:
	# don't make cropped_photos directories within other cropped_photos directories
	# and skip the base dir
	curr_dir = os.path.basename(v_dir) 
	if curr_dir == "cropped_photos" or curr_dir == "vc":
		continue
	print("Cleaning and cropping %s"%v_dir)
	# analyze_photos(v_dir)
