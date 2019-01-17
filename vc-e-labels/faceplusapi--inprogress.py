import urllib, json
import requests
import cv2
import glob
import time
import csv

file = open("ent-details.txt", "w") 
file.write("image_name"+ "\t"+"ethnicity"+ "\t"+"gender"+"\t"+"age"+"\t"+"beauty-female"+"\t"+"beauty-male"+"\t"+"\n")
for img_filename in glob.iglob('/Users/suprabhasomashekhar/Downloads/e/*/*.jpeg'):
	with open(img_filename, 'rb') as f:
		img_data = f.read()
		http_url="https://api-us.faceplusplus.com/facepp/v3/detect"
		path='/Users/suprabhasomashekhar/Downloads/'
		data={ "api_key": "8hvaF5CZTdoyaIOmxGNWbcPZzCTKJNUS","api_secret":"Pas8ETS8wSzEAFOeGh_r8frqGE3rhwU5","return_landmark":0, "return_attributes":"ethnicity,gender,age,beauty"}
		files= {"image_file": open(img_filename,'rb')}
		response=requests.post(http_url,data=data,files=files)
		time.sleep(2)
		req_con=response.content.decode('utf-8')
		print(req_con)
		index=img_filename.rfind("/")
		img_name=img_filename[index+1:]
		print(img_name)
		resp_dict = json.loads(req_con)
		try:
			file = open("ent-details.txt", "a") 
			file.write(img_name+ "\t"+str(resp_dict['faces'][0]["attributes"]["ethnicity"]["value"])+ "\t"+str(resp_dict['faces'][0]["attributes"]["gender"]["value"])+ "\t"+str(resp_dict['faces'][0]["attributes"]["age"]["value"])+ "\t"+str(resp_dict['faces'][0]["attributes"]["beauty"]["female_score"])+ "\t"+str(resp_dict['faces'][0]["attributes"]["beauty"]["male_score"])+ "\t"+"\n")
		except:
			continue
file.close()
