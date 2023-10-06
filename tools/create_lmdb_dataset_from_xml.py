import fire
import os
import lmdb
import cv2
import json
import xml.etree.ElementTree as eTree
import string
import sklearn.model_selection
import numpy as np
import glob

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False

    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH*imgW == 0:
        return False

    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for key, value in cache.items():
            txn.put(key, value)

def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test


def createDataset(input_dir, outputPath, characters, max_str_length):

    with open(characters, "r", encoding='utf-8') as jsonfile:
        characters_data = json.load(jsonfile)

    image_files = glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True)

    train_set, val_set = sklearn.model_selection.train_test_split(image_files, train_size=0.8, random_state=42)

    generate_files = glob.glob(os.path.join('../out/', '*.jpg'))
    # train_generate_set, val_generate_set = sklearn.model_selection.train_test_split(generate_files, train_size=0.8, random_state=42)
    train_generate_set=[]
    val_generate_set=[]

    print('start create lmdb dataset')

    sorted_chars = sorted(characters_data)
    # create lmdb
    outputPath_train = os.path.join(outputPath, 'training')
    outputPath_valid = os.path.join(outputPath, 'valid')
    #outputPath_test = os.path.join(outputPath, 'test')

    os.makedirs(outputPath_train, exist_ok=True)
    os.makedirs(outputPath_valid, exist_ok=True)
    #os.makedirs(outputPath_test, exist_ok=True)

    # create training dataset
    env = lmdb.open(outputPath_train, map_size=1099511627776 / 512)
    cache = {}
    cnt = 1

    for imagefilename in train_set:
        xml_file = imagefilename[:-3] + 'xml'
        utf_8_imagefilename = imagefilename.encode("utf-8", 'surrogateescape').decode('utf-8', 'replace')
        utf_8_labelfilename = xml_file.encode("utf-8", 'surrogateescape')

        print('processing file: ', utf_8_imagefilename)

        image = cv2.imread(utf_8_imagefilename)

        if image is None:
            n = np.fromfile(utf_8_imagefilename, np.uint8)
            image = cv2.imdecode(n, cv2.IMREAD_COLOR)

        _, width, _ = image.shape
        #with open(utf_8_imagefilename, 'rb') as f:
        #    imageBin = f.read()

        parser = eTree.XMLParser(encoding='utf-8')
        with open(utf_8_labelfilename, 'rb') as xml_file:
            xmltree = eTree.parse(xml_file, parser=parser).getroot()
            try:
                verified = xmltree.attrib['verified']
                if verified == 'yes':
                    verified = True
            except KeyError:
                verified = False

            for idx, object_iter in enumerate(xmltree.findall('object')):
                bndbox = object_iter.find("bndbox")
                sentence = object_iter.find('name').text

                if '\\n' in sentence:
                    sentence = sentence.replace('\\n', ' ')

                if any(c not in characters_data for c in sentence):
                    continue
                if max_str_length is not None and len(sentence) > max_str_length:
                    continue

                xmin = max(0, float(bndbox.find('xmin').text) - 10)
                ymin = float(bndbox.find('ymin').text)
                xmax = min(width, float(bndbox.find('xmax').text) + 10)
                ymax = float(bndbox.find('ymax').text)

                #box = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                imageBin = cv2.imencode('.jpg', image[round(ymin):round(ymax), round(xmin):round(xmax)])[1].tobytes()
                #imageBin = image[round(ymin):round(ymax), round(xmin):round(xmax)].tobytes()

                #dict_box = {"pt1": str(box[0]), "pt2": str(box[1]), "pt3": str(box[2]), "pt4": str(box[3])}
                #dict_data = {"image": imagefilename, "box": dict_box,  "sentence": sentence}

                imageKey = 'image-%09d'.encode() % cnt
                labelKey = 'label-%09d'.encode() % cnt
                cache[imageKey] = imageBin
                cache[labelKey] = sentence.encode()

                if cnt % 10000 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print('Written %d data' % (cnt))
                cnt = cnt + 1

    for imagefilename in train_generate_set:
        basefilename = os.path.basename(imagefilename)
        sentence = basefilename.split('_')[0]

        print('processing file: ', imagefilename)

        image = cv2.imread(imagefilename)

        if image is None:
            n = np.fromfile(imagefilename, np.uint8)
            image = cv2.imdecode(n, cv2.IMREAD_COLOR)

        imageBin = cv2.imencode('.jpg', image)[1].tobytes()

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = sentence.encode()

        if cnt % 10000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d data' % (cnt))
        cnt = cnt + 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created training dataset with %d samples' % nSamples)

    print('-' * 80)
    print('-' * 80)


    ### Test
    with env.begin(write=False) as txn:
        label_test = 1
        #label_key = 'label-%09d'.encode()
        #label = txn.get(label_key).decode('utf-8')
        #img_key = 'image-%09d'.encode()
        #imgbuf = txn.get(img_key)


    # create validation dataset
    env = lmdb.open(outputPath_valid, map_size=1099511627776 / 512)
    cache = {}
    cnt = 1

    for img_idx, imagefilename in enumerate(val_set):
        xml_file = imagefilename[:-3] + 'xml'
        utf_8_imagefilename = imagefilename.encode("utf-8", 'surrogateescape').decode('utf-8', 'replace')
        utf_8_labelfilename = xml_file.encode("utf-8", 'surrogateescape')

        print('processing file: ', utf_8_imagefilename)

        image = cv2.imread(utf_8_imagefilename)

        if image is None:
            n = np.fromfile(utf_8_imagefilename, np.uint8)
            image = cv2.imdecode(n, cv2.IMREAD_COLOR)

        _, width, _ = image.shape
        #with open(utf_8_imagefilename, 'rb') as f:
        #    imageBin = f.read()

        parser = eTree.XMLParser(encoding='utf-8')
        with open(utf_8_labelfilename, 'rb') as xml_file:
            xmltree = eTree.parse(xml_file, parser=parser).getroot()
            try:
                verified = xmltree.attrib['verified']
                if verified == 'yes':
                    verified = True
            except KeyError:
                verified = False

            for idx, object_iter in enumerate(xmltree.findall('object')):
                bndbox = object_iter.find("bndbox")
                sentence = object_iter.find('name').text

                if any(c not in characters_data for c in sentence):
                    continue
                if max_str_length is not None and len(sentence) > max_str_length:
                    continue

                xmin = max(0, float(bndbox.find('xmin').text) - 10)
                ymin = float(bndbox.find('ymin').text)
                xmax = min(width, float(bndbox.find('xmax').text) + 10)
                ymax = float(bndbox.find('ymax').text)

                box = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                imageBin = cv2.imencode('.jpg', image[round(ymin):round(ymax), round(xmin):round(xmax)])[1].tobytes()
                #imageBin = image[round(ymin):round(ymax), round(xmin):round(xmax)].tobytes()

                dict_box = {"pt1": str(box[0]), "pt2": str(box[1]), "pt3": str(box[2]), "pt4": str(box[3])}
                dict_data = {"image": imagefilename, "box": dict_box,  "sentence": sentence}

                imageKey = 'image-%09d'.encode() % cnt
                labelKey = 'label-%09d'.encode() % cnt
                cache[imageKey] = imageBin
                cache[labelKey] = sentence.encode()

                if cnt % 10000 == 0:

                    writeCache(env, cache)
                    cache = {}
                    print('Written %d data' % (cnt))
                cnt = cnt + 1

    for imagefilename in val_generate_set:
        basefilename = os.path.basename(imagefilename)
        sentence = basefilename.split('_')[0]

        print('processing file: ', imagefilename)

        image = cv2.imread(imagefilename)

        if image is None:
            n = np.fromfile(imagefilename, np.uint8)
            image = cv2.imdecode(n, cv2.IMREAD_COLOR)

        imageBin = cv2.imencode('.jpg', image)[1].tobytes()

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = sentence.encode()

        if cnt % 10000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d data' % (cnt))
        cnt = cnt + 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created valid dataset with %d samples' % nSamples)

if __name__ == '__main__':
    input_dir= "/home/OCR_id_passport_card"
    outputPath="/home/temp_data"
    characters="/home/parseq/tools/character_data.json"
    max_str_length=1000

    createDataset(input_dir, outputPath, characters, max_str_length)
    # fire.Fire(createDataset)/