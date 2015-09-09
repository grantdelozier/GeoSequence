import os
import sys
from random import shuffle
import shutil

args = sys.argv

#second argument should be directory being split
in_dir = args[1]
print "In dir: ", in_dir

ratio = .66

devtrain_outdir = "/Users/grant/devel/corpora/trconllf/xml/dev_trainsplit2"
devtest_outdir = "/Users/grant/devel/corpora/trconllf/xml/dev_testsplit2"

files = os.listdir(in_dir)
i = 0

shuffle(files)

for f in files:
	i += 1
	fp = os.path.join(in_dir, f)
	if float(i)/float(len(files)) < .66:
		shutil.copyfile(fp, os.path.join(devtrain_outdir, f))
	else:
		shutil.copyfile(fp, os.path.join(devtest_outdir, f))

print "finished"



