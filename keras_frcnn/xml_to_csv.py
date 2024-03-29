import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,                     
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
		     int(member[4][3].text),
                     member[0].text
                     )
            xml_list.append(value)
    column_name = ['path', 'xmin', 'ymin', 'xmax', 'ymax','class']
    xml_df = pd.DataFrame(xml_list,columns=column_name)
    return xml_df


def main():

    for folder in ['train_images','test_images']:
        image_path = os.path.join(os.getcwd(), (folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv((os.getcwd() +'/' folder + '_labels.txt'), index=None, header=False)
        print('Successfully converted xml to txt.')


main()
