import urllib, json
import requests
import cv2
import glob
import time
import csv

#file.write("faceId"+ "\t"+"faceTopDimension"+ "\t"+"faceLeftDimension"+ "\t"+"faceWidthDimension"+ "\t"+ "faceHeightDimension"+ "\t"+"Smile"+ "\t"+"pitch"+ "\t"+"roll"+ "\t"+"yaw"+ "\t"+"gender"+"\t"+"age"+"\t"+"moustache"+ "\t"+"beard"+ "\t"+"sideburns"+ "\t"+"glasses"+ "\t"+"anger"+ "\t"+"contempt"+ "\t"+"disgust"+ "\t"+"fear"+ "\t"+"hapiness"+ "\t"+"neutral"+ "\t"+"sadness"+ "\t"+"surprise"+ "\t"+"blurlevel"+ "\t"+"blurvalue"+ "\t"+"exposurelevel"+ "\t"+"exposurevalue"+ "\t"+"noiselevel"+ "\t"+"noisevalue"+ "\t"+"eymakeup"+ "\t"+"lipmakeup"+ "\t"+"foreheadoccluded"+ "\t"+"eyeoccluded"+ "\t"+"mouthoccluded"+ "\t"+"hair-bald"+ "\t"+"hair-invisible"+ "\t"+"img_name"+ "\t"+"\n")
file = open("ent-details.txt", "w") 
file.write("image_name"+ "\t"+"ethnicity"+ "\t"+"\n")
for img_filename in glob.iglob('/Users/suprabhasomashekhar/Downloads/*.jpg'):
	with open(img_filename, 'rb') as f:
		img_data = f.read()
		print(img_filename)
		http_url="https://api-us.faceplusplus.com/facepp/v3/detect"
		path='/Users/suprabhasomashekhar/Downloads/'
		data={ "api_key": "8hvaF5CZTdoyaIOmxGNWbcPZzCTKJNUS","api_secret":"Pas8ETS8wSzEAFOeGh_r8frqGE3rhwU5","return_landmark":0, "return_attributes":"ethnicity"}
		files= {"image_file": open(img_filename,'rb')}
		response=requests.post(http_url,data=data,files=files)
		time.sleep(2)
		req_con=response.content.decode('utf-8')
		print(req_con)
		print(req_con[0][0])
		file = open("out.csv", "ab")
		file.write(response.content)
		index=img_filename.rfind("/")
		img_name=img_filename[index+1:]
		