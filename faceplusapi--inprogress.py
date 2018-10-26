import urllib, json
import requests
import cv2
import glob
import time
import csv

file = open("ent-details.txt", "a") 
file.write("image_name"+ "\t"+"ethnicity"+ "\t"+"gender"+"\t"+"age"+"\t"+"\n")
for img_filename in glob.iglob('/Users/suprabhasomashekhar/Downloads/*.jpg'):
	with open(img_filename, 'rb') as f:
		img_data = f.read()
		print(img_filename)
		http_url="https://api-us.faceplusplus.com/facepp/v3/detect"
		path='/Users/suprabhasomashekhar/Downloads/'
		data={ "api_key": "8hvaF5CZTdoyaIOmxGNWbcPZzCTKJNUS","api_secret":"Pas8ETS8wSzEAFOeGh_r8frqGE3rhwU5","return_landmark":0, "return_attributes":"ethnicity,gender,age"}
		files= {"image_file": open(img_filename,'rb')}
		response=requests.post(http_url,data=data,files=files)
		time.sleep(2)
		req_con=response.content.decode('utf-8')
		print(req_con)
		index=img_filename.rfind("/")
		img_name=img_filename[index+1:]
		resp_dict = json.loads(req_con)
		file.write(img_name+ "\t"+str(resp_dict['faces'][0]["attributes"]["ethnicity"]["value"])+ "\t"+str(resp_dict['faces'][0]["attributes"]["gender"]["value"])+ "\t"+str(resp_dict['faces'][0]["attributes"]["age"]["value"])+ "\t"+"\n")
		
		
