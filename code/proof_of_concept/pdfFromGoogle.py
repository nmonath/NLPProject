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

browser = Firefox()

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

def getSearchAndDownloadPaperV2(textToSearch, fileNameToSave, extension):
	try:
			searchText = textToSearch.replace(' ', '+')
			urlToGet = 'https://www.google.com/search?q="' + searchText + '"+filetype%3Apdf'
			print urlToGet
			browser.get(urlToGet)
			page_source = browser.page_source
			start_position = page_source.find("<h3 class=\"r\"><a onmousedown",0)
			end_position = page_source.find("<h3 class=\"r\"><a onmousedown",start_position+10)
			first_answer = page_source[start_position:end_position]
			match = re.search(r'href.*?>',first_answer)
			url = first_answer[match.start()+6:match.end()-2]
			match2 = re.search(r'href.*?</a>',first_answer)
			descr = first_answer[match.end():match2.end()-4]
			if(descr[0:4] == '<em>'):
				descr = descr[4:len(descr)]
			words = len(textToSearch)
			if(words>30):
				words=30
			
			#if the url doesn't end in .pdf, try with the second one (only if it is highlighted), if it ends in .pdf in the second, download it
			
			if(url[len(url)-4:len(url)]!='.pdf'):
				print "\nThe following url doesn't end in pdf, looking for second result: " + url
				end_position2 = page_source.find("<h3 class=\"r\"><a onmousedown",end_position+10)
				second_answer = page_source[end_position:end_position2]
				match_second = re.search(r'href.*?>',second_answer)
				url_second = second_answer[match_second.start()+6:match_second.end()-2]
#				print "url second: " + url_second[len(url_second)-4:len(url_second)]
				match2_second = re.search(r'href.*?</a>',second_answer)
				descr_second = second_answer[match_second.end():match2_second.end()-4]
				if(descr_second[0:4] == '<em>' and url_second[len(url_second)-4:len(url_second)]==".pdf"):
					print "\nSecond seems to work better: " + url_second
					descr = descr_second[4:len(descr)]
					url=url_second
					words = len(textToSearch)
					if(words>30):
						words=30
				
			if(descr.lower()[0:words] == textToSearch.lower()[0:words]):
				print "\n\nMatch detected: " + fileNameToSave + "\n"
				print "\tText to search: " + textToSearch
				print "\n\tText found: " + descr.lower()
				print "\n\tUrl found: " + url
				urllib.urlretrieve(url, "../pdfdownloaded/" + fileNameToSave + extension)
			else: 
				print "\n\nNo match detected: \n" + fileNameToSave + "\n"
				print "\tText to search: " + textToSearch
				print "\n\tText found: " + descr.lower()
				print "\n\tUrl found: " + url
				urllib.urlretrieve(url, "../pdfdownloaded/" + fileNameToSave + "_possibly_wrong" + extension)
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
				getSearchAndDownloadPaperV2(curTit,curDoc, ".pdf")

browsePapersFile('../../data_sets/respapersim/documents.txt',26)
