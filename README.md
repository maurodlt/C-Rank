# C-Rank

C-Rank is an unsupervised keyphrase extraction algorithm that uses Concept Linking in order to improve its results.

It does not need external data to be inputted by the user other than the document to have its keyphrases extracted.

It is necessary to create an account to Babelfy (http://babelfy.org/login) as C-Rank uses its services. Then, your Babelfy key must be inserted in orde to C-Rank work properly.

## Installation
The following packages must be installed to use C-Rank:

networkx (https://networkx.github.io/): 
```
pip install networkx
```

nltk:(https://www.nltk.org/index.html)
```
sudo pip install -U nltk
```

pybabelfy: (https://github.com/aghie/pybabelfy)
```
sudo pip install pybabelfy
```

Then install C-Rank:
```
sudo pip install C-Rank
```


## Getting started
```
import CRank as cr

crank = cr.CRank(BABELFY_KEY, LIST_OF_INPUT_DOCUMENTS, OUTPUT_DIRECTORY)
#Example
#crank = cr.CRank("3ejklasd-a456-41ae-647f-0a1234546dd3", ['./document1.txt', './document2.txt'], './')
crank.keyphrasesExtraction()

crank.printKeyphrases()
```
## Functionalities
```
# all printing options 
printKeyphrases(self, nKeyphrases = 10, documentIndex=-1, showRanking = True, stem = False)

# save options to persist keyphrases in a single file (as in SemEval)
saveKeyphrasesSingleFile(self, fileName, nKeyphrases = 10, documentIndex=-1, showRanking = True, stem = False)

# save options to persist keyphrases in diferent files
saveKeyphrasesDiferentFiles(self, nKeyphrases = 10, documentIndex=-1, showRanking = True, stem = False)

# variables used in above functionalities
##nKeyphrases = number of kyphrases to print  |  nKeyphrases = 0 for all keyphrases
##documentIndex = index of document to print  |  documentIndex = -1 for all documents
##showRanking = show or not weight of keyphrases
##stem = stem or not keyphrases
##fileName = name of the file
```  
### Intermediate results and available variables
```
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
```
## Citation
Available soon

