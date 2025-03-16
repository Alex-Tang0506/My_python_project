import csv
toadd = ['Name', 'Portrait URL', 'Biography']
with open('D:\Tool\PythonProject\WebCrawler\Author_Info_Project\AL_AuthorInfo.csv',"r",) as infile:
    reader = list(csv.reader(infile))
    reader.insert(0,toadd)
with open('D:\Tool\PythonProject\WebCrawler\Author_Info_Project\AL_AuthorInfo.csv',"w",) as outfile:
    writer = csv.writer(outfile)
    for line in reader:
        writer.writerow(line)
infile.close()
outfile.close()