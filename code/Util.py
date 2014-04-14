def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"




def SVMLightWrite(targets, features, filename):
	[N,D]=features.shape;
	fid=file(filename,'w');
	for n in xrange(0, N):
		fid.write(str(targets[n]));
		for d in xrange(0,D):
			if (abs(features[n,d])>1e-3):
				fid.write(' ' + str(d+1) + ':' + str(features[n,d])) 
	fid.write('\n');
	fid.close()



