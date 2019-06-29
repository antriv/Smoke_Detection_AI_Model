import csv, json, sys
#if you are not using utf-8 files, remove the next line
#sys.setdefaultencoding("UTF-8") #set the encode to utf8
#check if you pass the input file and output file
#if sys.argv[1] is not None and sys.argv[2] is not None:
fileInput = "clip_1.json"
fileOutput = "clip_1.csv"
inputFile = open(fileInput) #open json file
outputFile = open(fileOutput, 'w') #load csv file
json_str = inputFile.read()
data = json.loads(json_str)
inputFile.close() #close the input file
output = csv.writer(outputFile) #create a csv.write
print(data[0][0].keys())
output.writerow(data[0][0].keys())  # header row

for block in data:
    #cnt=0
    for row in block:
        print(row.values())
        output.writerow(row.values())
        #cnt=cnt+1#values row

outputFile.close()
