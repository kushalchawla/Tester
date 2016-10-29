import numpy
import urllib
a = numpy.load("anno_list_mscoco_trainModelVal_m_RNN.npy")
for i in range(0,a.size):
	url = a[i]['url']
	file_name=a[i]['file_name']
	urllib.urlretrieve(url, "./downloaded_images_testing/" + file_name)
	print i
