#working on it 
import subprocess

def SRLParse(filenameOrigList, filenameConfig, folderDest):
	"""
		This function parses a list of files using Semantic Role Labeler of ClearNLP API. 
	"""
	command = "java com.clearnlp.nlp.engine.NLPDecode -z srl -c <filename> -i <filepath>"
	command = command.replace("<filename>", filenameConfig)
	for each fileName in filenameOrigList 
		execCmd = command.replace("<filepath>", fileName)
		process = subprocess.Popen(execCmd.split(), stdout=subprocess.PIPE)
		output = process.communicate()[0]


