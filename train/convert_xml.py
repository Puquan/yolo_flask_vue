import random
from config import init
import os

def extract_xml(path,img_path):
    result = img_path
    for line in open(path):
        if '<name>' in line:
            for i in range(len(labels)):
                if labels[i] in line:
                    labelid = str(i)
                    break
        if '<xmin>' in line:
            begin = line.find('<xmin>')
            end = line.find('</xmin>')
            xmin = line[begin+6:end]
        if '<ymin>' in line:
            begin = line.find('<ymin>')
            end = line.find('</ymin>')
            ymin = line[begin+6:end]
        if '<xmax>' in line:
            begin = line.find('<xmax>')
            end = line.find('</xmax>')
            xmax = line[begin+6:end]
        if '<ymax>' in line:
            begin = line.find('<ymax>')
            end = line.find('</ymax>')
            ymax = line[begin+6:end]
            result = result+' '+xmin+','+ymin+','+xmax+','+ymax+','+labelid
    return result


labels = init.XML.LABELS

paths = []
for path in (os.path.join(p, name) for p, _, names in os.walk(init.XML.INPUT_DIR) for name in names):
    paths.append(path)
random.shuffle(paths)


vp = init.XML.VP    #设置划分比例
mid = round(vp*len(paths)) #根据比例确定划分界限

ftrain = open(init.XML.TRAIN_TXT, 'w')

for img_path in paths[mid:]:
    # print(img_path)
    filename = img_path.split('/')[-1]
    pd = filename.split('.')[-1]
    if pd not in ("jpg", "png", "jpeg"):
        continue
    path = init.XML.XML_DIR + '/' + filename.split('.')[0] + '.xml'
    ftrain.write(extract_xml(path, img_path) + '\n')
ftrain.close()

ftest = open(init.XML.TEST_TXT, 'w')

for img_path in paths[0:mid]:
    filename = img_path.split('/')[-1]
    pd = filename.split('.')[-1]
    if pd not in ("jpg", "png", "jpeg"):
        continue
    path = init.XML.XML_DIR + '/' + filename.split('.')[0] + '.xml'
    ftest.write(extract_xml(path, img_path) + '\n')
ftest.close()