# based on https://github.com/monirome/AphasiaBank/blob/main/clean_transcriptions.ipynb
# based on https://github.com/Liting-Zhou/Aphasic_speech_recognition/blob/main/data_processing.py

import pylangacq as pla
import pandas as pd
import numpy as np
import sys
import re
import os

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#####################################################
# Load transcriptions

def read_chat_files(file_directory):
    try:
        ds = pla.read_chat(file_directory)
    except ValueError as e:
        # print(f"Error in file {file_directory}: {e}")
        return None
    files=ds.file_paths()

    #Time Mark (milisecond)
    cols = ['mark_start','mark_end']
    lst = []
    # revised by Liting
    for i in range(len(ds.utterances(participants="PAR"))):
        time_mark = ds.utterances(participants="PAR")[i].time_marks
        # Check if time_mark is None or has the correct structure
        if time_mark is None or len(time_mark) < 2:
            lst.append((np.nan, np.nan))  # Append NaN for both start and end if unavailable
        else:
            lst.append((time_mark[0], time_mark[1]))  # Append start and end as a tuple

    # original version
    # for i in range(len(ds.utterances(participants="PAR"))):
    #     lst.append(ds.utterances(participants="PAR")[i].time_marks)
    df = pd.DataFrame(lst, columns=cols) #create datrafame

    #patient transcription
    transcriptions=[]
    for i in range(len(ds.utterances(participants="PAR"))):
        transcriptions.append(ds.utterances(participants="PAR")[i].tiers['PAR'][:-16])
    df=df.assign(transcriptions=transcriptions)

    #patient information
    v_name=[] # add speaker name (Liting)
    v_sex=[]
    v_age=[]
    v_WAB_AQ=[]
    v_file_name=[]
    v_aphasia_type=[]
    for f in files: 
        ds2=pla.read_chat(f)
        participant=ds2.words(participants="PAR",by_utterances=True)
        size=len(participant)
        for j in range(size):
            header=ds2.headers()
            # Liting: Added speaker names
            try:
                name = header[0]['Participants']['PAR'].get('name', 'Unknown')
            except KeyError:
                name = 'Unknown'
            sex=header[0]['Participants']['PAR']['sex'] #sex information
            age=header[0]['Participants']['PAR']['age'] #age information
            WAB_AQ=header[0]['Participants']['PAR']['custom'] #WAB_AQ information
            aphasia_type=header[0]['Participants']['PAR']['group'] #fluency_speech information
            v_name.append(name)  # Liting: Added speaker names
            v_sex.append(sex)
            v_age.append(age)
            v_WAB_AQ.append(WAB_AQ)
            v_aphasia_type.append(aphasia_type)
            for k in range (len(ds2.file_paths())):
                v_file_name.append(ds2.file_paths()[k]) #file name 

    df = df.assign(name=v_name) #added by Liting
    df=df.assign(sex=v_sex)
    df=df.assign(age=v_age)
    df=df.assign(file=v_file_name)
    df=df.assign(WAB_AQ=v_WAB_AQ)
    df=df.assign(aphasia_type=v_aphasia_type)

    df['age'] = df['age'].str[:2] #remove the months
    path_len = len(file_directory) 
    df['file'] = df['file'].str[path_len:] #the number of str[] is the lenght of your directory
    df['file'] = df['file'].str[:-4]+'.wav' #file format

    return(df)

#####################################################
# Main function
def process_folder(folder_path):

    df = read_chat_files(folder_path)
    if df is None:
        return None

    #New metric: 'file_cut', where the start and duration of each transcription will be reflected
    #file_cut -> file_start_duration 

    v_file_cut=[]
    for i in range(len(df)):
        start=((pd.to_numeric(df['mark_start'][i]))/1000)
        duration=((pd.to_numeric(df['mark_end'][i]))-(pd.to_numeric(df['mark_start'][i])))/1000
        file=df['file'][i][:-4]
        file_cut=f"""{file}_{start}_{duration}.wav"""
        v_file_cut.append(file_cut)
    df=df.assign(file_cut=v_file_cut)

    #New metric: 'WAB_AQ_category', type of severity of the patient's aphasia
    #WAB_AQ_category -> aphasia type 

    df.loc[(pd.to_numeric(df['WAB_AQ'])>= 0) & (pd.to_numeric(df['WAB_AQ'])<=25), 'WAB_AQ_category'] = 'Very severe'
    df.loc[(pd.to_numeric(df['WAB_AQ'])> 25) & (pd.to_numeric(df['WAB_AQ'])<=50), 'WAB_AQ_category'] = 'Severe'
    df.loc[(pd.to_numeric(df['WAB_AQ'])> 50) & (pd.to_numeric(df['WAB_AQ'])<=75), 'WAB_AQ_category'] = 'Moderate'
    df.loc[(pd.to_numeric(df['WAB_AQ'])> 75) , 'WAB_AQ_category'] = 'Mild'
    df.loc[np.isnan(pd.to_numeric(df['WAB_AQ'])) , 'WAB_AQ_category'] = 'Unknown'

    #New metric: 'fluency_speech' is the patient's speech fluency based on the type of aphasia
    #fluency_speech -> speech fluency based on aphasia_type

    df.loc[((df['aphasia_type'])== 'Anomic') | ((df['aphasia_type'])== 'Conduction') | ((df['aphasia_type'])== 'Fluent')| ((df['aphasia_type'])== 'Wernicke')| ((df['aphasia_type'])== 'TransSensory'), 'fluency_speech'] = 'Fluent'
    df.loc[((df['aphasia_type'])== 'Broca') | ((df['aphasia_type'])== 'Global') | ((df['aphasia_type'])== 'TransMotor'), 'fluency_speech'] = 'Non Fluent'
    df.loc[((df['aphasia_type'])== 'NotAphasicByWAB') , 'fluency_speech'] = 'Unknown'

    # Add a new column with the folder name (Liting)
    folder_name = os.path.basename(os.path.normpath(folder_path))
    df = df.assign(folder_name=folder_name)

    #####################################################
    #clean dataset
    #The "filterWordsPhonetics" function changes the words written with phonemes for those that appear below between correctly written brackets

    def filterWordsPhonetic(string):
        while str(string).find("[:")>0:
            positionbracket = string.find("[:")
            if string[positionbracket -1] != " ": # If not an space before [:, add it
                string = string[:positionbracket] + " " + string[positionbracket:]
            
            list_words=string.split(" ")

            # Check if there are enough words
            if len(list_words) < 4:
                break  # Exit the loop if there aren't enough words

            try: 
                corchete_init=list_words.index("[:")
            except ValueError:
                try:
                    corchete_init=list_words.index("[::") # Sometimes they write [:: instead of [:
                except ValueError:
                    break
            
            if corchete_init > 0 and corchete_init + 1 < len(list_words):
                list_words[corchete_init-1] = list_words[corchete_init+1][0:-1] # change the bad word for good word
                del list_words[corchete_init:corchete_init+4] 
                string = " ".join(list_words)
            else:
                break  # Exit the loop if indices are invalid
        return string

    df['transcriptions'] = df['transcriptions'].map(filterWordsPhonetic)

    #A massive weird character cleanup is done
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\‡\„\$\^\/\//\0\↓\≠\↑]' 
    for i in range(len(df['transcriptions'])):
        df['transcriptions'][i]=re.sub(chars_to_ignore_regex,"", str(df["transcriptions"][i])).lower()

        df['transcriptions'][i]=re.sub('\+<', ' ',str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('&=laughs','<LAU>',str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('&=chuckles','<LAU>',str(df["transcriptions"][i]))
        
        df['transcriptions'][i]=re.sub('\&=\w*:\w*_\w*','',str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('[\[].*?[\]]','',str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\&=[a-z]*:[a-z]*','',str(df["transcriptions"][i])) #quita las trasncripciones tipo &=word:word

        df['transcriptions'][i]=re.sub('&-.([a-z]*)','F',str(df["transcriptions"][i])) #quita las trasncripciones tipo &-word
        df['transcriptions'][i]=re.sub('&.([a-z]*)','F',str(df["transcriptions"][i])) #quita las trasncripciones tipo &word
        
        df['transcriptions'][i]=re.sub('\+', ' ',str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('[/&=] *','S',str(df["transcriptions"][i])) #fix bug, Liting

        df['transcriptions'][i]=re.sub('dada@b',"F", str(df["transcriptions"][i]))    
        df['transcriptions'][i]=re.sub('_',' ',df["transcriptions"][i] )
        df['transcriptions'][i]=re.sub('\x15',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\)',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\(',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[  gra',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('   ', ' ', str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('  ', ' ', str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub(r'[0-9]',"", df["transcriptions"][i])
        df['transcriptions'][i]=re.sub('xn', '', str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub(r"  "," ", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('blublublubluhb',"S", df["transcriptions"][i])    
        df['transcriptions'][i]=re.sub('www',"", str(df["transcriptions"][i]))

    for i in range(len(df['transcriptions'])):
        df['transcriptions'][i]=re.sub('<seeing>',"seeing", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<seeing them>',"seeing", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<wanna>',"wanna", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<that>',"that", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<yeah>',"yeah", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<okay>',"okay", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<just>',"just", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<with>',"with", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<raindrops>',"raindrops", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<and>',"and", str(df["transcriptions"][i]))

    for i in range(len(df['transcriptions'])):
        df['transcriptions'][i]=re.sub('\[<spn',"S", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[<',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[ gr',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[ g',"", str(df["transcriptions"][i]))
        
        df['transcriptions'][i]=re.sub('\[ jar',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[ ja',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[ j',"", str(df["transcriptions"][i]))
        
        df['transcriptions'][i]=re.sub('\[ exc',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[ ex',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[ e',"", str(df["transcriptions"][i]))
        
        df['transcriptions'][i]=re.sub('\[ es',"", str(df["transcriptions"][i]))
        
        df['transcriptions'][i]=re.sub('\[ ci',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\[ c',"", str(df["transcriptions"][i]))
        
        df['transcriptions'][i]=re.sub('gram]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('pn]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('nk]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('srgcpro]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('belt]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('sr]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('suk]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('\]',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('/snan/s',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<xxx>',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('<xxx',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('xxx>',"", str(df["transcriptions"][i]))
        df['transcriptions'][i]=re.sub('@',"", str(df["transcriptions"][i]))

    df['transcriptions'] = df['transcriptions'].apply(lambda s: ' '.join(i for i in s.split(' ') if i != 'x'))
    df['transcriptions'] = df['transcriptions'].apply(lambda s: ' '.join(i for i in s.split(' ') if i != 'xx'))
    df['transcriptions'] = df['transcriptions'].apply(lambda s: ' '.join(i for i in s.split(' ') if i != 'xxx'))

    p='é|æ|ɑ|ɔ|ɕ|ç|ḏ|ḍ|ð|ə|ɚ|ɛ|ɝ|ḡ|ʰ|ḥ|ḫ|ẖ|ɪ|ỉ|ɨ|ʲ|ǰ|ḳ|ḵ|ḷ|ɬ|ɫ|ŋ|ṇ|ɲ|ɴ|ŏ|ɸ|θ|p̅|þ|ɹ|ɾ|ʀ|ʁ|ṛ|š|ś|ṣ|ʃ|ṭ|ṯ|ʨ|tʂ|ʊ|ŭ|ü|ʌ|ɣ|ʍ|χ|ʸ|ʎ|ẓ|ž|ʒ|’|‘|ʔ|ʕ|∬|↫'
    df=df.loc[~df['transcriptions'].str.contains(p, regex=True)]  #remove words with phonemes
    df = df.reset_index(drop=True)

    # Added by Liting
    # 1. remove rows where 'mark_start' or 'mark_end' columns are empty
    df.dropna(subset=['mark_start', 'mark_end'], inplace=True)
    # 2. remove everything inside <> or every word that starts with < or ends with >
    df['transcriptions'] = df['transcriptions'].str.replace(r'<.*?>', '', regex=True)
    df['transcriptions'] = df['transcriptions'].str.replace(r'\s*<\w*|\w*>\s*', ' ', regex=True)
    # 3. remove single capital letter like F and S
    df['transcriptions'] = df['transcriptions'].apply(
        lambda x: ' '.join([word for word in x.split() if word != 'F' and word != 'S'])
    )
    # 4. remove single word 'xn'
    df['transcriptions'] = df['transcriptions'].apply(lambda x: re.sub(r'\s*xn\s*', '', str(x)))

    df=df[~df["transcriptions"].isnull()] #remove nulls
    df["transcriptions"] = df["transcriptions"].str.rstrip() #remove blanks at the beginning 
    df["transcriptions"] = df["transcriptions"].str.lstrip() #remove blanks at the end
    df['transcriptions'] = df['transcriptions'].str.replace(r'\s+', ' ', regex=True) # replace multiple spaces with a single space

    # save the dataset in a folder for using in the following scipts
    output_file = os.path.join(folder_path, 'clean_dataset.csv')
    df.to_csv(output_file, index=False, header=True)
    print(f"Processed data saved to: {output_file}")

#####################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_script.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    process_folder(folder_path)
