# Running Final Retrieval Experiment

import Retrieval
import Features
import Word2VecExecuter


Features.USE_LEMMA = True
Features.USE_DEP_TAGS = False
Features.USE_POS_TAGS = False
Features.USE_ARG_LABELS = False
Features.USE_MEMORY_MAP = False
Features.FREP = Features.FeatureRepresentation.HASH
Features.FTYPE = Features.FeatureType.TFIDF



print ('\n'+ '%' *80 + '\n') 
# Words
Features.FUNIT = Features.FeatureUnits.WORD
Features.DisplayConfiguration()
Output = Retrieval.RunExhaustive(Features, '../data_sets/research_papers', TopK=5, train_test='train')

print ('\n'+ '%' *80 + '\n') 
print ('\n'+ '%' *80 + '\n') 

# Words & DP
Features.FUNIT = Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS
Features.DisplayConfiguration()
Output = Retrieval.RunExhaustive(Features, '../data_sets/research_papers', TopK=5, train_test='train')
print ('\n'+ '%' *80 + '\n') 
print ('\n'+ '%' *80 + '\n') 

# Words & PA
Features.FUNIT = Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT
Features.DisplayConfiguration()
Output = Retrieval.RunExhaustive(Features, '../data_sets/research_papers', TopK=5, train_test='train')
print ('\n'+ '%' *80 + '\n') 
print ('\n'+ '%' *80 + '\n') 


# ALL
Features.FUNIT = Features.FeatureUnits.ALL
Features.DisplayConfiguration()
Output = Retrieval.RunExhaustive(Features, '../data_sets/research_papers', TopK=5, train_test='train')

print ('\n'+ '%' *80 + '\n') 
print ('\n'+ '%' *80 + '\n') 
model = Word2VecExecuter.Word2VecGetModel("/Users/nmonath/Downloads/GoogleNews-vectors-negative300.bin")

print("Embeddings KD Tree")
Output = Retrieval.RunEKD('../data_sets/research_papers', model, train_test='train',TopK=5)

print ('\n'+ '%' *80 + '\n') 
print ('\n'+ '%' *80 + '\n') 

print("PAD")
Output = Retrieval.RunPAD('../data_sets/research_papers', model, train_test='train',TopK=5)

print ('\n'+ '%' *80 + '\n') 
