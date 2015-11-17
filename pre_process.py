from collections import Counter
import sys
from os import listdir,makedirs
from os.path import isfile,join,exists
from collections import defaultdict
import math
import re

stop_words = ['a','about','above','across','after','again','against','all','almost','alone','along','already','also','although','always','among','an','and','another','any','anybody','anyone','anything','anywhere','are','area','areas','around','as','ask','asked','asking','asks','at','away','b','back','backed','backing','backs','be','became','because','become','becomes','been','before','began','behind','being','beings','best','better','between','big','both','but','by','c','came','can','cannot','case','cases','certain','certainly','clear','clearly','come','could','d','did','differ','different','differently','do','does','done','down','down','downed','downing','downs','during','e','each','early','either','end','ended','ending','ends','enough','even','evenly','ever','every','everybody','everyone','everything','everywhere','f','face','faces','fact','facts','far','felt','few','find','finds','first','for','four','from','full','fully','further','furthered','furthering','furthers','g','gave','general','generally','get','gets','give','given','gives','go','going','good','goods','got','great','greater','greatest','group','grouped','grouping','groups','h','had','has','have','having','he','her','here','herself','high','high','high','higher','highest','him','himself','his','how','however','i','if','important','in','interest','interested','interesting','interests','into','is','it','its','itself','j','just','k','keep','keeps','kind','knew','know','known','knows','l','large','largely','last','later','latest','least','less','let','lets','like','likely','long','longer','longest','m','made','make','making','man','many','may','me','member','members','men','might','more','most','mostly','mr','mrs','much','must','my','myself','n','necessary','need','needed','needing','needs','never','new','new','newer','newest','next','no','nobody','non','noone','not','nothing','now','nowhere','number','numbers','o','of','off','often','old','older','oldest','on','once','one','only','open','opened','opening','opens','or','order','ordered','ordering','orders','other','others','our','out','over','p','part','parted','parting','parts','per','perhaps','place','places','point','pointed','pointing','points','possible','present','presented','presenting','presents','problem','problems','put','puts','q','quite','r','rather','really','right','right','room','rooms','s','said','same','saw','say','says','second','seconds','see','seem','seemed','seeming','seems','sees','several','shall','she','should','show','showed','showing','shows','side','sides','since','small','smaller','smallest','so','some','somebody','someone','something','somewhere','state','states','still','still','such','sure','t','take','taken','than','that','the','their','them','then','there','therefore','these','they','thing','things','think','thinks','this','those','though','thought','thoughts','three','through','thus','to','today','together','too','took','toward','turn','turned','turning','turns','two','u','under','until','up','upon','us','use','used','uses','v','very','w','want','wanted','wanting','wants','was','way','ways','we','well','wells','went','were','what','when','where','whether','which','while','who','whole','whose','why','will','with','within','without','work','worked','working','works','would','x','y','year','years','yet','you','young','younger','youngest','your','yours','z']


class Trie:
    """
    Implement a trie with insert, search, and startsWith methods.
    """
    def __init__(self):
        self.root = defaultdict()

    def insert(self, word):
        current = self.root
        for letter in word:
            current = current.setdefault(letter, {})
        current.setdefault("_end")

    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        if "_end" in current:
            return True
        return False

def enumerate_words(current,cstr,result):

    if current == None:
        result.append(''.join([x for x in cstr[:-1]]))
        return
        #print current.keys()
    for k,v in current.iteritems():
            cstr.append(k)
            enumerate_words(v,cstr,result)
            cstr.pop()

"""
Purpose:- Function takes file and checks whether it should be used for processing or not
If file contains line which contains word "subject:" and "re:" in same line then we should ignore it.
Altenatively,Any postings that have a "re:" as the first part of their "subject:" line is a reply and is eliminated.

Return Value:- Return True if it is valid posting else False
"""

def is_original_posting(fname):
	found_subject_tag = False
	found_lines_tag = False
	with open(fname,'r') as f:
		for line in f:
			line = line.lower()
			
			idx1 = line.find("subject:")
			idx2 = line.find("re:")
			idx3 = line.find("lines:")

			if idx3 >= 0:
				found_lines_tag = True
				if not found_subject_tag:
					return False

			if 0 == idx1:
				found_subject_tag = True

			if 0 == idx1 and idx2 >=0:
				return False
	return True

"""
Purpose:- Function takes file and return all the text after the tag "subject:"
Return Value:- Returns list where each element is line of the file
"""

def extract_text(fname):
	base_idx = 8
	result = []
	should_include = False
	req_lines = 0
	clines = 0
	with open(fname,'r') as f:
		for line in f:

			line = line.lower()
			idx1 = line.find("subject:")
			
			if req_lines == 0 and line.find("lines:")>=0:
				tokens = line.split(":")
				try:
					req_lines = int(tokens[1].strip())
				except:
					req_lines = 0
				should_include = True
			elif idx1 >=0:
				result.append(line[idx1 + base_idx:])
			elif should_include:
				if req_lines > clines:
					result.append(line)
				clines = clines+1
	
	return result
		
"""
Purpose:- Function that takes a string as input and removes all non-ascii characters
Return Value:- Returns string that only contains acii characters
"""
def remove_nonascii(line):
	return ''.join([i if ord(i) < 128 else '' for i in line])

"""
Purpose:- Function that takes string as input and removes all non-alphanumeric characters
Return Value:- Returns string that only contains alpha numeric characters
"""
def remove_non_alpha_numeric(line):
	return re.sub('[^0-9a-z]+', ' ',line)

"""
Purpose:- Entry point of pre-processing for line. Takes string as input and pre-processes it by applying above mentioned function
Return Value:- Returns a list with all valid words in string
"""
def pre_process(line):

	result = []
	line = remove_nonascii(line)
	line = remove_non_alpha_numeric(line)
	tokens = line.split(" ")
	for tk in tokens:
		tk = tk.strip()
		if tk.isdigit() or len(tk) == 0 or tk in stop_words:
			continue
		result.append(tk)
	return result
	
"""
Purpose:- Entry point of pre-processing for file.
Return Value:- Returns a list with valid words in file
"""
def process(fname):
	original_posting = is_original_posting(fname)
	if not original_posting:
		return
	
	result = []
	text = extract_text(fname)
	for line in text:
		line = pre_process(line)
		if line and len(line)>0:
			result = result + line
	
	return result

"""
Purpose:- Returns ngram representation of given token.
Parameters:- string,ngram
Return Value:- list containing ngrams of size ngrams (character-based not word-based)
"""
def get_ngram_representation(tokens,ngrams):
	text = " ".join([x for x in tokens])
	return [text[i-ngrams:i+1] for i,char in enumerate(text)][ngrams:]

"""
Purpose:- Writing features(words) to file
"""
def dump_features_to_file(base_dir,fname,features):
	if not exists(base_dir):
		makedirs(base_dir)

	with open(join(base_dir,fname),'w') as f:
		for word in features:
			f.write(word+"\n")

def dump_label_to_file(base_dir,fname,word_dict):
	if not exists(base_dir):
		makedirs(base_dir)

	result = []
	for k,v in word_dict.iteritems():
		result.append((v[0],v[1],v[2]))
	result.sort()

	with open(join(base_dir,fname),'w') as f:
		for i in range(len(result)):
			f.write(str(result[i][0]) + "," + result[i][1] + "_" + str(result[i][2]) +"\n")

"""
Purpose:- Writing Vector representation of word to file so that it can be reterieved later for ckustering
Parameter:- base_dir -> Directory to write the output
			fname -> File to write the output
			word_dict -> dictionary of (word,frequency) to be dumped into file
"""
def dump_class_to_file(base_dir,fname,word_dict):
	if not exists(base_dir):
		makedirs(base_dir)

	result = []
	for k,v in word_dict.iteritems():
		result.append((v[0],v[1]))
	result.sort()

	with open(join(base_dir,fname),'w') as f:
		for i in range(len(result)):
			f.write(str(result[i][0]) + "," + result[i][1]+"\n")

def normalize(word_dict,word_frequency,tag,clean_files,words):
	result = {}
	for k,v in word_dict.iteritems():
		sum_sq = 0
		for fv,value in v:
			idf = math.log((clean_files*1.0)/len(word_frequency[words[fv]][tag]))
			tf_idf = (value * idf)
			sum_sq = sum_sq + tf_idf**2
		
		sum_sq = sum_sq**0.5
		result_vect = []
		csum = 0
		for fv,value in v:
			idf = math.log((clean_files*1.0)/len(word_frequency[words[fv]][tag]))
			tf_idf = (value * idf)
			normalized_freq = (tf_idf*1.)/sum_sq
			result_vect.append((fv,normalized_freq))
		
		result[k] = result_vect
	return result

def dump_to_input_file(base_dir,fname,word_dict,word_frequency,tag,clean_files,words,use_tfidf=False):

	if not exists(base_dir):
		makedirs(base_dir)

	if use_tfidf:
		word_dict = normalize(word_dict,word_frequency,tag,clean_files,words)
	with open(join(base_dir,fname),'w') as f:
		for k,v in word_dict.iteritems():
			v.sort()
			for fv,value in v:
				f.write(str(k)+","+str(fv)+","+str(value)+"\n")

def process_tokens(t_trie,word_frequency,all_keys,tag,token,object_id):

	t_trie.insert(token)
	if token not in word_frequency:
		word_frequency[token] = {}
		for key in all_keys:
			word_frequency[token][key] = set()
	if object_id not in word_frequency[token][tag]:
		word_frequency[token][tag].add(object_id)

def remove_unecessary_terms(feature_vector,word_frequency,tag,threshold,total_files,lower_limit):

	tfeature_vector = []
	for word in feature_vector:
		if len(word_frequency[word][tag]) > lower_limit and len(word_frequency[word][tag]) < threshold*total_files:
			tfeature_vector.append(word)
	return tfeature_vector

"""
Purpose:- Generate vectors corresponding to Bag of Words and Ngrams approach
Parameters:- base_folder -> Directory from where to read the files
			 ngrams -> list specifying different length of ngrams
			 verbose -> write intermediate output to stdout
"""
def generate_feature_space(base_folder,ngrams,output_dir):
	
	file_bow_dict = {}
	file_ngram_dict = {}

	all_keys = ['bag']
	for ngram in ngrams:
		all_keys.append('char'+str(ngram))
	
	feature_space = {}
	feature_space_trie = {}
	for ngram in ngrams:
		feature_space[ngram] = []
		feature_space_trie[ngram] = Trie()

	class_map = {}
	feature_vector_trie = Trie()
	feature_vector = []
	word_frequency = {}

	files = listdir(base_folder)
	clean_files = 0
	for data_file in files:
		if not isfile(data_file):
			data_file_path = join(base_folder,data_file)
			postings = listdir(data_file_path)
			for input_file in postings:
				file_path = join(join(base_folder,data_file),input_file)
				file_tokens = process(file_path)
				if file_tokens and len(file_tokens) > 0:
					file_bow_dict[file_path] = file_tokens
					
					for token in file_tokens:
						process_tokens(feature_vector_trie,word_frequency,all_keys,'bag',token,clean_files)

					ngrams_for_file = {}
					for ngram in ngrams:
						ngrams_for_file[ngram] = get_ngram_representation(file_tokens,ngram-1)
						for token in ngrams_for_file[ngram]:
							process_tokens(feature_space_trie[ngram],word_frequency,all_keys,'char'+str(ngram),token,clean_files)

					file_ngram_dict[file_path] = ngrams_for_file
					class_map[file_path] = (clean_files,data_file,input_file)
					clean_files = clean_files + 1

	print "Clean files " + str(clean_files)
	lower_limit = 2
	threshold = 0.90
	enumerate_words(feature_vector_trie.root,[],feature_vector)
	feature_vector = remove_unecessary_terms(feature_vector,word_frequency,'bag',threshold,clean_files,lower_limit)

	for ngram in ngrams:
		required_key = 'char'+str(ngram)
		enumerate_words(feature_space_trie[ngram].root,[],feature_space[ngram])
		feature_space[ngram] = remove_unecessary_terms(feature_space[ngram],word_frequency,'char'+str(ngram),threshold,clean_files,lower_limit)

	dump_class_to_file(output_dir,'newsgroups.class',class_map)
	dump_label_to_file(output_dir,'newsgroups.rlabel',class_map)
	return (file_bow_dict,feature_vector,file_ngram_dict,feature_space,class_map,word_frequency,clean_files)	

"""
Purpose:- Convert each file to its Bag of words and Ngram representation
Parameters:- base_folder -> Directory from where to read the files
			 output_base_dir -> Directory where to write output file
			 feature_vector -> consists of valid unique words across all files
			 feature_space -> dictionary whose key is ngram and value is all unqiue ngrams across all file in base_folder
"""

def create_feature_vector(base_folder,output_base_dir,ngrams,file_bow_dict,feature_vector,file_ngram_dict,feature_space,class_map,word_frequency,clean_files,use_tfidf):
	
	if not exists(output_base_dir):
		makedirs(output_base_dir)		

	fv_dict = {}
	for i,word in enumerate(feature_vector):
		fv_dict[word] = i
	
	dump_features_to_file(output_base_dir,'bag.clabel',feature_vector)

	output_ngram_dict = {}
	for ngram in ngrams:
		output_ngram_dict[ngram] = {}
		for i,word in enumerate(feature_space[ngram]):
			output_ngram_dict[ngram][word] = i
		dump_features_to_file(output_base_dir,'char'+str(ngram)+'.clabel',feature_space[ngram])		

	output_feature_vector = {}
	for fname,feature_vector_list in file_bow_dict.iteritems():
		filename = class_map[fname][0]
		output_feature_vector[filename] = []
		word_count = Counter(feature_vector_list)
		
		normalize_factor = 0
		for k,v in word_count.iteritems():
			if k in fv_dict:
				output_feature_vector[filename].append((fv_dict[k],v))

	#Normalize and then write
	dump_to_input_file(output_base_dir,'bag.csv',output_feature_vector,word_frequency,'bag',clean_files,feature_vector,use_tfidf)	

	output_feature_space = {}
	for ngram in ngrams:
		output_feature_space[ngram] = {}

	for fname,ngram_dict in file_ngram_dict.iteritems():
		filename = class_map[fname][0]
		
		for ngram,values in ngram_dict.iteritems():
			if filename not in output_feature_space[ngram]:
				output_feature_space[ngram][filename] = []
			ngram_count = Counter(values)
			normalize_factor = 0
			for k,v in ngram_count.iteritems():
				if k in output_ngram_dict[ngram]:
					output_feature_space[ngram][filename].append((output_ngram_dict[ngram][k],v))

	for k,v in output_feature_space.iteritems():
		dump_to_input_file(output_base_dir,"char"+str(k)+".csv",output_feature_space[k],word_frequency,'char'+str(k),clean_files,feature_space[k],use_tfidf)
	
total_args = len(sys.argv)
base_folder = '20_newsgroups'
ngrams = [3,5,7]
output_dir = '.'
use_tfidf = True

if total_args >= 2:
	if sys.argv[1].find("use_tfidf")>=0:
		tokens = sys.argv[1].split("=")
		if 2 == len(tokens) and "0" == tokens[1]:
			use_tfidf = False

file_bow_dict,feature_vector,file_ngram_dict,feature_space,class_map,word_frequency,clean_files = generate_feature_space(base_folder,ngrams,output_dir)
create_feature_vector(base_folder,output_dir,ngrams,file_bow_dict,feature_vector,file_ngram_dict,feature_space,class_map,word_frequency,clean_files,use_tfidf)
