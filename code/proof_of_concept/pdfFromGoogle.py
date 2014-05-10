#import urllib2
import urllib
import time
import random 
from pygoogle import pygoogle

def getSearchAndDownloadPaper(textToSearch, fileNameToSave):
	g = pygoogle(textToSearch + ' filetype:pdf')
	g.pages = 1
	try:
		pdfUrl = g.get_urls()[0]
		urllib.urlretrieve(pdfUrl, "../pdfdownloaded/" + fileNameToSave)
		time.sleep(random.randint(30,60));

	except IndexError: 
		print fileNameToSave + " " + textToSearch
		time.sleep(180);

def browsePapersFile(pathToFile, startWith):
	print pathToFile
	f = open(pathToFile, 'r')
	curDoc = '0'
	curTit = 'none'
	out = f.readlines()
	print 'read'
	for line in out:
#		print line
		if line[0:3] == 'DOC': 
			curDoc = line[4:len(line)-2]
		if  line[0:3] == 'TIT': 
			curTit = line[4:len(line)-2]
			if(int(curDoc)>startWith):
				print curDoc
				getSearchAndDownloadPaper(curTit,curDoc + ".pdf")
browsePapersFile('../../data_sets/respapersim/documents.txt',26)