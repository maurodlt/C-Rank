import string
import networkx as nx
import re
from nltk import word_tokenize, pos_tag
from pybabelfy.babelfy import *
from CRank.porterStemmer import PorterStemmer
from math import log

class CRank():

	def __init__(self, BabelfyKey, inputFiles, outputDirectory = './', dist = 3, language = 'EN'):
		#init variables
		self.key = BabelfyKey
		self.inputFiles = inputFiles
		self.outputDirectory = outputDirectory
		self.lang = language
		self.distance = dist
		self.graphName = []
		self.splitted_text = []
		self.dictionary = []
		self.dictionaryCode = []
		self.weight = []
		self.paragraphs_annotations = []
		self.paragraphs_text = []
		self.paragraphs_code = []
		self.graphs = []
		self.graphs2 = []
		self.keyPhrases = []

		if self.outputDirectory[-1] != '/':
			self.outputDirectory = self.outputDirectory + '/'

	def keyphrasesExtraction(self):
		count = 1
		for file in self.inputFiles:
			print('Processing Document ' + str(count) + '/' + str(len(self.inputFiles)))
			self.graphName.append(file.split('/')[-1])

			############################### Candidates Keyphrases Extraction ################################
			#parse document
			text = self.openFile(file)
			st = self.parseText(text)
			self.splitted_text.append(st)

			#concepts link
			try:
				pa, pt, pc  = self.babelfy(self.lang, self.key, st)
			except ValueError:
				print('Error during Babelfication! Verify if you have Babelcoins available at http://babelfy.org/home')
				print(str(count -1) + " documents had their keyphrases succesfully extracted, which are available in the 'keyPhrases' variable")
				break
			self.paragraphs_annotations.append(pa)
			self.paragraphs_text.append(pt)
			self.paragraphs_code.append(pc)
			d, dc, w = self.createDicts(pt, pc)
			self.dictionary.append(d)
			self.dictionaryCode.append(dc)
			self.weight.append(w)

			############################### Weight Candidates ################################
			#generate graphs
			g, g2 = self.createGraph(w, pc, self.distance)
			self.graphs.append(g)
			self.graphs2.append(g2)

			#rank graph
			keyphrases = self.nodeRank(g)

			#heuristics application
			keyphrases = self.heuristic1(keyphrases, dc)
			keyphrases, lenght, totalWords = self.heuristic2(keyphrases, pc)
			keyphrases = self.heuristic3(keyphrases, dc)
			g, g2, keyphrases = self.heuristic4(g, g2, totalWords, keyphrases, pc)

			############################### Keyphrase Extraction ################################
			#Compound Candidates Identification & Re-rank candidates
			keyphrases = self.keyPhrasesCompilation(keyphrases,g,g2,dc,lenght,totalWords)
			self.keyPhrases.append(keyphrases)
			count += 1

	def stemKeyphrase(self, keyphrase):
		stemmer=PorterStemmer()
		parsedPhrase = ''
		for wr in keyphrase.split(' '):
			for w,i in zip(wr.split('-'),range(len(wr.split('-')))):
				if w != '':
					parsedWord = stemmer.stem(w.lower(), 0, len(w)-1)
					if i >= 1:
						parsedPhrase = parsedPhrase + '-' + parsedWord
					else:
						parsedPhrase = parsedPhrase + parsedWord

			parsedPhrase = parsedPhrase + ' '
		parsedPhrase = parsedPhrase[:len(parsedPhrase)-1]
		parsedPhrase = parsedPhrase.replace('  ',' ')

		return parsedPhrase

	## nKeyphrases = 0 for all keyphrases
	## documentIndex = -1 for all documents
	def printKeyphrases(self, nKeyphrases = 10, documentIndex=-1, showRanking = True, stem = False):
		if documentIndex == -1:
			for d in range(0,len(self.keyPhrases)):
				print('Document: ' + str(d))
				if nKeyphrases <= 0 or nKeyphrases >= len(self.keyPhrases[d]):
					nKeyphrases = len(self.keyPhrases[d])
				for k in range(0, nKeyphrases):
					keyphrase = ""
					if stem:
						keyphrase = self.stemKeyphrase(self.keyPhrases[d][k][0])
					else:
						keyphrase = self.keyPhrases[d][k][0]

					if showRanking:
						print('\t' + str(keyphrase) + ', ' + str(self.keyPhrases[d][k][1]))
					else:
						print('\t' + keyphrase)

		elif documentIndex >= 0 and documentIndex < len(self.keyPhrases):
			print('Document: ' + str(documentIndex))
			if nKeyphrases <= 0 or nKeyphrases >= len(self.keyPhrases[documentIndex]):
				nKeyphrases = len(self.keyPhrases[documentIndex])
			for k in range(0, nKeyphrases):
				keyphrase = ""
				if stem:
					keyphrase = self.stemKeyphrase(self.keyPhrases[documentIndex][k][0])
				else:
					keyphrase = self.keyPhrases[documentIndex][k][0]

				if showRanking:
					print('\t' + str(keyphrase) + ', ' + str(self.keyPhrases[documentIndex][k][1]))
				else:
					print('\t' + keyphrase)

		else:
			print('Insert a valid document index')

	def saveKeyphrasesSingleFile(self, fileName, nKeyphrases = 10, documentIndex=-1, showRanking = True, stem = False):
		file = open(self.outputDirectory + fileName,"w+")
		if documentIndex == -1:
			for d in range(0,len(self.keyPhrases)):
				file.write(str(self.graphName[d]) + ':')
				if nKeyphrases <= 0 or nKeyphrases >= len(self.keyPhrases[d]):
					nKeyphrases = len(self.keyPhrases[d])
				for k in range(0, nKeyphrases):
					keyphrase = ""
					if stem:
						keyphrase = self.stemKeyphrase(self.keyPhrases[d][k][0])
					else:
						keyphrase = self.keyPhrases[d][k][0]

					if showRanking:
						file.write(str(keyphrase) + ',' + str(self.keyPhrases[d][k][1]) + ';')
					else:
						file.write(keyphrase + ';')
				file.write('\n')

		elif documentIndex >= 0 and documentIndex < len(self.keyPhrases):
			file.write(str(self.graphName[documentIndex]) + ':')
			if nKeyphrases <= 0 or nKeyphrases >= len(self.keyPhrases[documentIndex]):
				nKeyphrases = len(self.keyPhrases[documentIndex])
			for k in range(0, nKeyphrases):
				keyphrase = ""
				if stem:
					keyphrase = self.stemKeyphrase(self.keyPhrases[documentIndex][k][0])
				else:
					keyphrase = self.keyPhrases[documentIndex][k][0]

				if showRanking:
					file.write(str(keyphrase) + ',' + str(self.keyPhrases[documentIndex][k][1]) + ';')
				else:
					file.write(keyphrase + ';')

		else:
			print('Insert a valid document index')

		file.close()


	def saveKeyphrasesDiferentFiles(self, nKeyphrases = 10, documentIndex=-1, showRanking = True, stem = False):
		if documentIndex == -1:
			for d in range(0,len(self.keyPhrases)):
				file = open(self.outputDirectory + self.graphName[d] + '.keyphrases',"w+")
				if nKeyphrases <= 0 or nKeyphrases >= len(self.keyPhrases[d]):
					nKeyphrases = len(self.keyPhrases[d])
				for k in range(0, nKeyphrases):
					keyphrase = ""
					if stem:
						keyphrase = self.stemKeyphrase(self.keyPhrases[d][k][0])
					else:
						keyphrase = self.keyPhrases[d][k][0]

					if showRanking:
						file.write(str(keyphrase) + ',' + str(self.keyPhrases[d][k][1]) + '\n')
					else:
						file.write(keyphrase + '\n')
				file.close()

		elif documentIndex >= 0 and documentIndex < len(self.keyPhrases):
			file = open(self.outputDirectory + self.graphName[documentIndex] + '.keyphrases',"w+")
			if nKeyphrases <= 0 or nKeyphrases >= len(self.keyPhrases[documentIndex]):
				nKeyphrases = len(self.keyPhrases[documentIndex])
			for k in range(0, nKeyphrases):
				keyphrase = ""
				if stem:
					keyphrase = self.stemKeyphrase(self.keyPhrases[documentIndex][k][0])
				else:
					keyphrase = self.keyPhrases[documentIndex][k][0]

				if showRanking:
					file.write(str(keyphrase) + ',' + str(self.keyPhrases[documentIndex][k][1]) + '\n')
				else:
					file.write(keyphrase + '\n')
			file.close()

		else:
			print('Insert a valid document index')

	def heuristic4(self, g, g2, totalWords, keyphrases, pc):
		#use only words that appeared at first 18% of the text in long documents
		firstWords = []

		#verify if document is long
		if totalWords > 1000:
			threshold = int(totalWords * .18)
		else:
			return g, g2, keyphrases

		#identify words at the beggining of the text
		for p in pc:
			for w in p:
				if len(firstWords) < threshold:
					firstWords.append(w)
				else:
					break
		newKeyphrases = []
		for k in keyphrases:
			if k[0] in firstWords:
				newKeyphrases.append(k)
		keyphrases = newKeyphrases

		#remove words that were not at the beggining of the document
		nodesToRemove = []
		for n in g:
			inKeyphrases = False
			for k in keyphrases:
				if k[0] == n:
					inKeyphrases = True
			if inKeyphrases == False:
				nodesToRemove.append(n)
		for n in nodesToRemove:
			g.remove_node(n)
			g2.remove_node(n)

		return g, g2, keyphrases


	def heuristic3(self, keyphrases, dc):
		#re-weight mult-term keyphrases
		keyphrases = [[k[0], (k[1]**(1/(len(dc[k[0]].split(' ')))))] for k in keyphrases]
		keyphrases = sorted(keyphrases, key=lambda kv: (kv[1], kv[0]), reverse=True)
		return keyphrases

	def heuristic2(self, keyphrases, pc):
		#excludes last 87% keyphrases of long documents
		totalWords = sum([len(i) for i in pc])

		if totalWords > 1000:
			lenght = int(.13*len(keyphrases))
		else:
			lenght = 1

		summation = sum([v[1] for v in keyphrases[:lenght]])
		keyphrases = [[k[0], k[1]/summation] for k in keyphrases[:lenght]]
		return keyphrases, lenght, totalWords

	def heuristic1(self, keyphrases, dc):
		#Exclude words different from nouns, verbs and adjectives
		new_keyphrases = []
		for k in keyphrases:
			words = dc[k[0]]
			tokens = word_tokenize(words)
			notDesiredTags = False

			for w in pos_tag(tokens):
				if w[1][0] != 'N' and w[1][0] != 'J' and w[1][0] != 'V':
					notDesiredTags = True
			if notDesiredTags:
				exclude = [k[0], 0]
				new_keyphrases.append(exclude)
			else:
				new_keyphrases.append(k)
		new_keyphrases = sorted(new_keyphrases, key=lambda kv: (kv[1], kv[0]), reverse=True)
		return new_keyphrases

	# open and close file
	def openFile(self, fileName):
	    file = open(fileName,"r")
	    text = file.read()
	    file.close()
	    return text

	#parse and split text in chuncks of at most 3000 characters
	def parseText(self, text):
		#remove special characters
		punctuationToRemove = string.punctuation.replace('!','').replace('.','').replace('?','').replace('-','').replace(',','')
		translator = str.maketrans('', '', punctuationToRemove)
		parsedText = text.translate(translator)
		#remove numbers
		parsedText = re.sub(r'[0-9]+', '', parsedText)
		#remove double spaces
		parsedText = re.sub(r'  ', ' ', parsedText)
		#remove non-printable characters
		parsedText = "".join(filter(lambda x: x in string.printable, parsedText))

		#split text in chuncks of at most 5000 characters
		punctuation = ['.','?','!']
		splitted_text = []
		splitted_text.append("")
		for line in parsedText.splitlines():
			if len(splitted_text[-1] + line) < 5000 and splitted_text[-1][-1:] not in punctuation or len(splitted_text[-1] + line) <= 3000:
				splitted_text[-1] = splitted_text[-1] + ' ' + line
			else:
				splitted_text.append(line)
		translator = str.maketrans('', '', "?!.")
		for l in splitted_text:
			l = l.translate(translator)
		return splitted_text


	def frag(self, semantic_annotation, input_text):
	    start = semantic_annotation.char_fragment_start()
	    end = semantic_annotation.char_fragment_end()
	    return input_text[start:end+1]


	def babelfy(self, lang, key, splitted_text):
		babelapi = Babelfy()
		#bn = BabelNet(key)
		paragraphs_annotations = []
		paragraphs_text = []
		paragraphs_code = []

		count = 0
		for paragraph in splitted_text: #annotate each paragraph
			words_annotations = []
			words_text = []
			words_code = []
			semantic_annotations = babelapi.disambiguate(paragraph,lang,key,match="EXACT_MATCHING",cands="TOP",mcs="ON",anntype="ALL")

			#exclude unused annotations (single words of multiword expressions)
			for semantic_annotation in semantic_annotations:
			    if len(words_annotations) == 0 or words_annotations[-1].char_fragment_end() < semantic_annotation.char_fragment_start():
			        words_annotations.append(semantic_annotation)
			        words_text.append(self.frag(semantic_annotation,paragraph))
			        words_code.append(semantic_annotation.babel_synset_id())

			    elif words_annotations[-1].char_fragment_start() == semantic_annotation.char_fragment_start():
			        del words_annotations[-1]
			        words_annotations.append(semantic_annotation)
			        del words_text[-1]
			        words_text.append(self.frag(semantic_annotation,paragraph))
			        del words_code[-1]
			        words_code.append(semantic_annotation.babel_synset_id())


			paragraphs_annotations.append(words_annotations)
			paragraphs_text.append(words_text)
			paragraphs_code.append(words_code)
			count = count + 1
			print(str(count) + '/' + str(len(splitted_text)))
		return paragraphs_annotations, paragraphs_text, paragraphs_code


	def saveText(self, file, paragraphs_code):
		file = open(file,"w+")
		for p in paragraphs_code:
			for w in p:
				file.write(w + "#")
			file.write("\n")
		file.close()


	#Create the following Dicts
	def createDicts(self, paragraphs_text, paragraphs_code):
		###		dictionary[word] = code		 	###
		###		dictionaryCode[code] = word		###
		###		weight[codigo] = weight			###
		dictionary={}
		weight={}
		dictionaryCode={}
		for paragraph, codes in zip(paragraphs_text, paragraphs_code):
			for word, code in zip(paragraph, codes):
				if code not in weight:
					weight[code] = 1
				else:
					weight[code] = weight[code] + 1

				if word not in dictionary:
					dictionary[word] = code
				if code not in dictionaryCode:
					dictionaryCode[code] = word
		return dictionary, dictionaryCode, weight



	def createGraph(self, peso, paragraphs_code, dist):
		g = nx.DiGraph() #indirect Graph
		g2 = nx.DiGraph() #direct Graph

		#calc the weight of each vertice
		for code, weight in peso.items():
			g.add_node(code, peso=weight)
			g2.add_node(code, peso=weight)

		#create and weight edges
		for line in paragraphs_code:
			i = 0
			for word in line:
				i = i + 1
				j = 0
				for word2 in line:
					j = j + 1
					if j - i < dist and j - i > 0: #indirect edges
						if g.has_edge(word, word2):
							g[word][word2]['weight'] += 1 - log(j-i,dist)
						else:
							if word != word2:
								g.add_edge(word, word2, weight=float(1 - log(j-i,dist)))
					if j - i == 1: #direct edges
						if g2.has_edge(word, word2):
							g2[word][word2]['weight'] += 1
						else:
							if word != word2:
								g2.add_edge(word, word2, weight=1)
		return g, g2


	def nodeRank(self, g):
		r = nx.degree_centrality(g)
		rank = {}
		for p in r:
			rank[p] = r[str(p)]
		rank = sorted(rank.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		return rank

	def saveKeyWords(self, keyWords, graphName, resultFile):
	    file = open(resultFile,"w+")
	    stemmer=PorterStemmer()
	    for words, name in zip(keyWords, graphName):
	        file.write(name + " : ")
	        for word, weight, weight2 in words[:15]:
	            parsedPhrase = ''
	            for wr in word.split(' '):
	                for w,i in zip(wr.split('-'),range(len(wr.split('-')))):
	                    if w != '':
	                        parsedWord = stemmer.stem(w.lower(), 0, len(w)-1)
	                        if i >= 1:
	                            parsedPhrase = parsedPhrase + '-' + parsedWord
	                        else:
	                            parsedPhrase = parsedPhrase + parsedWord

	                parsedPhrase = parsedPhrase + ' '
	            parsedPhrase = parsedPhrase[:len(parsedPhrase)-1]
	            parsedPhrase = parsedPhrase.replace('  ',' ')

	            if(word != words[0][0]):
	                file.write("," + parsedPhrase)
	            else:
	                file.write(parsedPhrase)
	        file.write("\n")
	    file.close()
	    return

	def takeSecond(self, elem):
		return elem[1]

	def keyPhrasesCompilation(self, keyWords, g, g2, dictionaryCode,lenght,totalWords):
	    keyphrases_dict = {code : value for code, value in keyWords}

	    coumpound_keyphrases = []

	    #identify compound keyphrases
	    for k in keyWords[:lenght]:
	        for k2 in keyWords[:lenght]:
	            if g2.has_edge(k[0], k2[0]) and g2[k[0]][k2[0]]['weight'] >= int(totalWords / 1000) + 2: #verify occurrences
	                if g2.has_edge(k2[0], k[0]) == False or g2[k2[0]][k[0]]['weight'] < g2[k[0]][k2[0]]['weight']:
	                    weight = g.out_degree(k[0], weight='weight') + g.in_degree(k2[0], weight='weight') #normalization factor | w(Out(i)) + w(In(j))
	                    phrase = [k[0] + ',' + k2[0], g[k[0]][k2[0]]['weight'] / weight] #weight compound keyphrase  |  NIE_{i,j}
	                    if phrase not in coumpound_keyphrases:
	                        coumpound_keyphrases.append(phrase)

	    coumpound_keyphrases = sorted(coumpound_keyphrases, key=lambda kv: (kv[1], kv[0]), reverse=True)
	    keyphrases_weight = [t[1] for t in coumpound_keyphrases]
	    keyphrases_weight_norm = [float(i)/sum(keyphrases_weight) for i in keyphrases_weight]  #normalize NIE   |   NIE_{i,j} / (sum_all(NIE))
	    keyphrases = [t for t in coumpound_keyphrases]

	    for kp, n in zip(keyphrases, keyphrases_weight_norm):
	        codes = kp[0].split(',')
	        #rank keyphrases
	        kp[1] = ((keyphrases_dict[codes[0]] + keyphrases_dict[codes[1]])) * n   # CC_{i,j}
	        kp[0] = dictionaryCode[codes[0]] + ' ' + dictionaryCode[codes[1]]

	    soma = sum([v[1] for v in keyphrases])
	    keyphrases = [[k[0], k[1]/soma] for k in keyphrases]   #NCC_{i,j}
	    keywords = [[dictionaryCode[k[0]], k[1]] for k in keyWords]
	    merged = keyphrases[:6] + keywords #FWC U NCC_{1:6}

	    merged.sort(key=self.takeSecond, reverse=True)

	    return merged
