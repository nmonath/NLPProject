#import urllib2
import urllib
from pygoogle import pygoogle
import re 
import time
import random
from pygoogle import pygoogle
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver import Firefox
from contextlib import closing
import traceback

def getSearchAndDownloadPaper(textToSearch, fileNameToSave):
	g = pygoogle(textToSearch + ' filetype:pdf')
	g.pages = 1
	try:
		pdfUrl = g.get_urls()[0]
		urllib.urlretrieve(pdfUrl, "../pdfdownloaded/" + fileNameToSave)
		time.sleep(random.randint(30,60))
	except IndexError:
		print fileNameToSave + " " + textToSearch
		time.sleep(180);

def getSearchAndDownloadPaperV2(textToSearch, fileNameToSave):
	try:
		with closing(Firefox()) as browser:
			searchText = textToSearch.replace(' ', '+')
			urlToGet = 'https://www.google.com/search?q="' + searchText + '"+filetype%3Apdf'
			browser.get(urlToGet)
			page_source = browser.page_source
			start_position = page_source.find("<h3 class=\"r\"><a onmousedown",0)
			end_position = page_source.find("<h3 class=\"r\"><a onmousedown",start_position+10)
			first_answer = page_source[start_position:end_position]
			match = re.search(r'href.*?>',first_answer)
			url = first_answer[match.start()+6:match.end()-2]
			match2 = re.search(r'href.*?</a>',first_answer)
			descr = first_answer[match.end():match2.end()-4]
			words = len(textToSearch)
			if(words>30):
				words=30
			if(descr.lower()[0:words] == textToSearch.lower()[0:words]):
				print "\n\nMatch detected: " + fileNameToSave + "\n"
				print "\tText to search: " + textToSearch
				print "\n\tText found: " + match2.lower()
				print "\n\tUrl found: " + url
				urllib.urlretrieve(url, "../pdfdownloaded/" + fileNameToSave)
			else: 
				print "\n\nNo match detected: \n" + fileNameToSave + "\n"
				print "\tText to search: " + textToSearch
				print "\n\tText found: " + descr.lower()
				print "\n\tUrl found: " + url
				urllib.urlretrieve(url, "../pdfdownloaded/" + fileNameToSave + "_wrong")
	except:
		print "\n ERROR FOR: " + fileNameToSave + "\n" + traceback.format_exc() + "\n"

def browsePapersFile(pathToFile, startWith):
	print pathToFile
	f = open(pathToFile, 'r')
	curDoc = '0'
	curTit = 'none'
	out = f.readlines()
	print 'read'
	for line in out:
		# print line
		if line[0:3] == 'DOC':
			curDoc = line[4:len(line)-2]
		if line[0:3] == 'TIT':
			curTit = line[4:len(line)-2]
			if(int(curDoc)>startWith):
				print curDoc
				getSearchAndDownloadPaperV2(curTit,curDoc + ".pdf")

browsePapersFile('../../data_sets/respapersim/documents.txt',0)
