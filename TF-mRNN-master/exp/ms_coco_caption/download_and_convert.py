import pickle
import Image
import sys
import urllib
#proxies = {}
#proxies['http'] = "http://prateek:medellin@172.16.114.251:3128"
#@https_proxy = "https://prateek:medellin@172.16.114.251:3128"
#export https_proxy
def processImage(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print "Cant load", infile
        sys.exit(1)
    count = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            
            count += 1
            im.seek(im.tell() + 1)
    except EOFError:
        pass # end of sequence


    try:
        im = Image.open(infile)
    except IOError:
        print "Cant load", infile
        sys.exit(1)
    count = count/10
    i = 0
    mypalette = im.getpalette()

    try:

        while i < count*10:
            im.putpalette(mypalette)
            if(i%count == 0):
                
                new_im = Image.new("RGBA", im.size)
                new_im.paste(im)
                img_name = infile.split('/')[1] + '/' + infile.split('/')[2].split('.')[0]
                new_im.save(img_name+str(i)+'.jpg')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence


d = {}
f = open("tgif-v1.0.tsv","r")
for line in f:
    l = line.split("\t")
    
    l2 = l[0].split("/")[-1]
    print l2
    d[l2] = l[1]
    urllib.urlretrieve(l[0], "./downloaded_gifs/" + l2)#,proxies = proxies)
    processImage('./downloaded_gifs/' + l2 )


with open('anno_list_mscoco_trainModelVal_m_RNN.pickle', 'wb') as handle:
  pickle.dump(d, handle)


#https_proxy="http://www.someproxy.com:3128"