#-*- coding:utf-8 -*-
import os
import codecs
from operator import itemgetter
def doc_sellect(il,InputDir,lexicon,geofile,wikifile,gazfile,translation):
    eng_word = {}
    thre_h = 1000
    f = open('/nas/data/m1/lud2/LORELEI/document_selection/key_words/incidentvocab_il12')
    for oneline in f:
        if oneline.strip() not in eng_word:
            eng_word[oneline.strip().lower()] = set([])
    f.close()

    fil = open(lexicon)
    for oneline in fil:
        if oneline.strip().split('\t')[-1].lower() in eng_word:
            print(oneline.strip().split('\t')[0].lower(),oneline.strip().split('\t')[-1])
            
            eng_word[oneline.strip().split('\t')[-1].lower()].add(oneline.strip().split('\t')[0].lower())
            
    fil.close()
    '''
    ftr = open(translation)
    for oneline in ftr:
        if oneline.strip().split('\t')[-1].lower() in eng_word:
            print(oneline.strip().split('\t')[0].lower(),oneline.strip().split('\t')[-1])

            eng_word[oneline.strip().split('\t')[-1].lower()].add(oneline.strip().split('\t')[0].lower())

    ftr.close()
    '''
    fgaz = open(gazfile)
    for oneline in fgaz:
        if oneline.strip().split('\t')[0].lower() in eng_word:
            eng_word[oneline.strip().split('\t')[0].lower()].add(oneline.strip().split('\t')[-1].lower())
            print(oneline.strip().split('\t')[-1])
    fgaz.close()


    fwiki = open(wikifile)
    for oneline in fwiki:
        if oneline.strip().split(':::')[1].lower() in eng_word:
            eng_word[oneline.strip().split(':::')[1].lower()].add(oneline.strip().split(':::')[0].lower())
            print(oneline.strip().split(':::')[0])
    fwiki.close()
    
    fgeo = open(geofile)
    for oneline in fgeo:
        if oneline.strip().split('\t')[-1].lower() in eng_word:
            eng_word[oneline.strip().split('\t')[-1].lower()].add(oneline.strip().split('\t')[0].lower())
            print(oneline.strip().split('\t')[0],oneline.strip().split('\t')[-1].lower())
    fgeo.close()

    print('xxx','\t'.join(eng_word['guam']))
    doc_count = []
    index = 0
    for onefile in os.listdir(InputDir):
        if '_RF_' in onefile:
            continue
        index+=1
#        if index>10000:
#            break
        key_word_count = {}
        f = open(os.path.join(InputDir,onefile))
        article = f.read().lower()
        f.close()
        #if list(eng_word['addis ababa'])[0] in article:
        #    print 'xxx',list(eng_word['addis ababa'])[0]
        all_count = 0
        context = []
        for one_word in eng_word:
            key_word_count[one_word] = 0
            for one_il in eng_word[one_word]:
                key_word_count[one_word] += article.count(one_il)
            #key_word_count[one_word] = article.count(one_word)
            if key_word_count[one_word]>0:
                context.append(one_word)
                #context.append(str(article.count(one_word)))
                context.append(str(key_word_count[one_word]))
                #        print one_word,article.count(one_word)
                #context += '%s-%s\t'%(one_word,article.count(one_word))
                #all_count += article.count(one_word)
                all_count += key_word_count[one_word]
                #print onefile, all_count
                #print all_count
                #if all_count>100:
                #    print all_count
                #    print context
        if len(article.split())>thre_h:
            continue
        doc_count.append([onefile,all_count,context])
    order_doc_count = sorted(doc_count,key=itemgetter(1),reverse=True)
    return order_doc_count



if __name__=='__main__':
    il = 'ilo'
#    InputDir = '/nas/data/m1/LORELEI_Data/LDC_raw_data/LDC2017E29_LORELEI_IL6_Incident_Language_Pack_for_Year_2_Eval_V1.1/set0/data/monolingual_text/rsd_tokenized'
    InputDir = '/nas/data/m1/liny9/lorelei2019/eval/data/ilocano/set0/rsd'
    #InputDir = '/data/m1/lud2/LORELEI/data/VOA/Oromo/rsd'
    lexicon = '/nas/data/m1/lud2/LORELEI/cleaned_multilingual_data/src/Panlex-Lexicon-Extractor/data/lexicons/%s_eng_lexicon.txt'%il
    geofile = '/nas/data/m1/lud2/LORELEI/cleaned_multilingual_data/geonames/ilo_geonames.txt'
    wikifile = '/nas/data/m1/lud2/LORELEI/cleaned_multilingual_data/wikiname/name_pair_ilo'
    gazfile = '/nas/data/m1/lud2/LORELEI/yr4/docs/il12/IL12_dictionary.txt'
    translation = '/data/m1/liny9/lorelei/resource/orm_anno_trans.txt'
    order_doc_count = doc_sellect(il,InputDir,lexicon,geofile,wikifile,gazfile,translation)
    f_out = open('%s_topic_set0.csv'%il,'w')

    for one in order_doc_count[:5000]:
#        print one[0],one[1],'\t'.join(one[2])
        f_out.write('%s\t%s\t%s\n'%(one[0],one[1],'\t'.join(one[2])))

    f_out.close()
    
