from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.meta_cat import MetaCAT


def identify_Sysmptoms(sentence):

    # Load the vocab model you downloaded
    vocab= Vocab.load('/mnt/c/Users/hmasu/Downloads/vocab.dat')

    # Load the cdb model you downloaded
    cdb = CDB.load('/mnt/c/Users/hmasu/Downloads/cdb-medmen-v1.dat') 

    # create cat
    cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
    doc_spacy = cat(sentence)

    # Or to get an array of entities, this will return much more information
    #and usually easier to use unless you know a lot about spaCy
    doc = cat.get_entities(sentence)
    print(doc)

def identify_Sysmptoms_meta_cat(sentence):
    # Assume we have a CDB and Vocab object from before
    # Download the mc_status model from the models section below and unzip it

    vocab= Vocab.load('/mnt/c/Users/hmasu/Downloads/vocab.dat')
    cdb = CDB.load('/mnt/c/Users/hmasu/Downloads/cdb-medmen-v1.dat') 

    mc_status = MetaCAT.load("/mnt/c/Users/hmasu/Downloads/mc_status")
    cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab, meta_cats=[mc_status])

    # Now annotate a document, it will have the meta annotation 'status'
    doc = cat.get_entities(sentence)
    print(doc['entities'])
    for d in doc['entities']:
        print(d[0])
   
   



def main():
    value = input("Please enter a string:\n")
    if value != '':
        identify_Sysmptoms_meta_cat(value)

if __name__ == "__main__":
    main()