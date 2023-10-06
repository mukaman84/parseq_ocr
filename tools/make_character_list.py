import glob
import os
import xml.etree.ElementTree as eTree
import json
import fire


def make_character_list(data_dir, output_json):
    label_files = glob.glob(os.path.join(data_dir, '**','*.xml'), recursive=True)
    alphabet_sample = []
    # get statistics of letters in data
    for xml_file in label_files:
        utf_8_labelfilename = xml_file.encode("utf-8", 'surrogateescape')
        xml_parser = eTree.XMLParser(encoding='utf-8')
        with open(utf_8_labelfilename, 'rb') as xml_file:
            xmltree = eTree.parse(xml_file, parser=xml_parser).getroot()
            try:
                verified = xmltree.attrib['verified']
                if verified == 'yes':
                    verified = True
            except KeyError:
                verified = False
            for idx, object_iter in enumerate(xmltree.findall('object')):
                bndbox = object_iter.find("bndbox")
                sentence = object_iter.find('name').text

                for w in list(sentence):
                    if w is not '\t':
                        alphabet_sample.append(w)

    #alphabet_recog
    #letter_stat = Counter(alphabet_sample)
    #letter_stat_lexico = sorted(letter_stat.items())
    #alphabet_recog = []
    #for letter, freq in letter_stat_lexico:
        #if freq > 9 and '가' <= letter and '힣' >= letter:
        #    alphabet_recog.append(letter)

    recognizer_alphabet = sorted(set(alphabet_sample))

    # os.makedirs(os.path.dirname(json_alphabet), exist_ok=True)
    with open(output_json, "w", encoding='utf-8') as jsonfile:
        json.dump(recognizer_alphabet, jsonfile)
    # letter_stat_freq = letter_stat.most_common()

if __name__ == '__main__':
    fire.Fire(make_character_list)