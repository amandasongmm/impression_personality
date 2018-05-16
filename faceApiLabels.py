import urllib, json
import requests
import cv2
import glob
import time
import csv
i=0
file = open("ent-details.txt", "w") 
file.write("faceId"+ "\t"+"faceTopDimension"+ "\t"+"faceLeftDimension"+ "\t"+"faceWidthDimension"+ "\t"+ "faceHeightDimension"+ "\t"+"Smile"+ "\t"+"pitch"+ "\t"+"roll"+ "\t"+"yaw"+ "\t"+"gender"+"\t"+"age"+"\t"+"moustache"+ "\t"+"beard"+ "\t"+"sideburns"+ "\t"+"glasses"+ "\t"+"anger"+ "\t"+"contempt"+ "\t"+"disgust"+ "\t"+"fear"+ "\t"+"hapiness"+ "\t"+"neutral"+ "\t"+"sadness"+ "\t"+"surprise"+ "\t"+"blurlevel"+ "\t"+"blurvalue"+ "\t"+"exposurelevel"+ "\t"+"exposurevalue"+ "\t"+"noiselevel"+ "\t"+"noisevalue"+ "\t"+"eymakeup"+ "\t"+"lipmakeup"+ "\t"+"foreheadoccluded"+ "\t"+"eyeoccluded"+ "\t"+"mouthoccluded"+ "\t"+"hair-bald"+ "\t"+"hair-invisible"+ "\t"+"img_name"+ "\t"+"\n")
for img_filename in glob.iglob('/Users/suprabhasomashekhar/Downloads/e_no_mask/*.jpg'):
	if cv2.imread(img_filename) is None:
		continue
	with open(img_filename, 'rb') as f:
		img_data = f.read()
		subscription_key = 'bba8d4fc85574e34a3ce28923f8b6665'
		header = {'Content-Type': 'application/octet-stream','Ocp-Apim-Subscription-Key': subscription_key,}
		params = urllib.parse.urlencode({'returnFaceId': 'true','returnFaceLandmarks': 'false','returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',})
		api_url = "https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect?%s"%params
		r = requests.post(api_url,params=params,headers=header,data=img_data)
		print (r.json())
		
		x = r.json()
		file = open("ent-details.txt", "a") 
		print()
		index=img_filename.rfind("/")
		img_name=img_filename[index+1:]
		try:
			file.write(x[0]['faceId']+ "\t"+str(x[0]['faceRectangle']['top'])+ "\t"+str(x[0]['faceRectangle']['left'])+ "\t"+str(x[0]['faceRectangle']['width'])+ "\t"+str(x[0]['faceRectangle']['height'])+ "\t"+ str(x[0]['faceAttributes']['smile'])+"\t"+str(x[0]['faceAttributes']['headPose']['pitch'])+"\t"+str(x[0]['faceAttributes']['headPose']['roll'])+"\t"+str(x[0]['faceAttributes']['headPose']['yaw'])+"\t"+x[0]['faceAttributes']['gender']+"\t"+str(x[0]['faceAttributes']['age'])+"\t"+str(x[0]['faceAttributes']['facialHair']['moustache'])+"\t"+str(x[0]['faceAttributes']['facialHair']['beard'])+"\t"+str(x[0]['faceAttributes']['facialHair']['sideburns'])+"\t"+str(x[0]['faceAttributes']['glasses'])+"\t"+str(x[0]['faceAttributes']['emotion']['anger'])+"\t"+str(x[0]['faceAttributes']['emotion']['contempt'])+"\t"+str(x[0]['faceAttributes']['emotion']['disgust'])+"\t"+str(x[0]['faceAttributes']['emotion']['fear'])+"\t"+str(x[0]['faceAttributes']['emotion']['happiness'])+"\t"+str(x[0]['faceAttributes']['emotion']['neutral'])+"\t"+str(x[0]['faceAttributes']['emotion']['sadness'])+"\t"+str(x[0]['faceAttributes']['emotion']['surprise'])+"\t"+str(x[0]['faceAttributes']['blur']['blurLevel'])+"\t"+str(x[0]['faceAttributes']['blur']['value'])+"\t"+str(x[0]['faceAttributes']['exposure']['exposureLevel'])+"\t"+str(x[0]['faceAttributes']['exposure']['value'])+"\t"+str(x[0]['faceAttributes']['noise']['noiseLevel'])+"\t"+str(x[0]['faceAttributes']['noise']['value'])+"\t"+str(x[0]['faceAttributes']['makeup']['eyeMakeup'])+"\t"+str(x[0]['faceAttributes']['makeup']['lipMakeup'])+"\t"+str(x[0]['faceAttributes']['occlusion']['foreheadOccluded'])+"\t"+str(x[0]['faceAttributes']['occlusion']['eyeOccluded'])+"\t"+str(x[0]['faceAttributes']['occlusion']['mouthOccluded'])+"\t"+str(x[0]['faceAttributes']['hair']['bald'])+"\t"+str(x[0]['faceAttributes']['hair']['invisible'])+"\t"+img_name+"\n")
		except:
			if i%20==0 :
				time.sleep(60)
			i=i+1
			continue
		if i%20==0 :
			time.sleep(60)
		i=i+1