import numpy as np
annotationFile = 'C:/Users/aande355/Desktop/TrafficSigns/signDatabasePublicFramesOnly/allAnnotations.csv'
rows = open(annotationFile).read().strip().split("\n")[1:]
tags = []
for row in rows:
    args = row.strip().split(";")
    tags.append(args[1])

tags = np.unique(tags)
f = open("./LISAsignnames.csv",'w')
f.write('ClassId,SignName\n')
for i,tag in enumerate(tags):
    f.write(str(i)+","+tag+'\n')
f.close()
