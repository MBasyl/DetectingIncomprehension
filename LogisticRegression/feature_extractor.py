from lexicalrichness import LexicalRichness
import textstat 
import complexity_measures.taassc  as T # from https://github.com/kristopherkyle/TAASSC
import re
import feature_choose as F 

def feature_extractor(lines, LIST_F, file):
        """Applies complexity measures on file with dialogueID<TAB>label<TAB>sentence-pair"""
        complexity= []
        for line in lines:
                utt_score = []
                uttL=line.split("\t")
                utts = uttL[2:]

                # superficial, lexical, syntactic measures
                for text in utts:
                        n_sent = textstat.sentence_count(text)
                        n_wrd = textstat.lexicon_count(text, removepunct=True) 
                        n_ch = textstat.char_count(text, ignore_spaces=True)
                        wrd_nchr = round(n_ch / n_wrd, 2)
                        dale_read = textstat.dale_chall_readability_score(text) 
                        n_polysyl = textstat.polysyllabcount(text) 
                        lex = LexicalRichness(text) # LEXICAL DIVERSITY (NOT DENSTY)
                        try: 
                          ttr = round(lex.ttr, 2) 
                          mtld = round(lex.mtld(threshold=0.72),2)
                        except ZeroDivisionError:
                          ttr = 0
                          mtld =0

                        complexity_score = [n_sent,wrd_nchr,dale_read,n_polysyl,ttr,mtld]
                        syn = T.BTR_Analysis(text, T.index_list, T.cat_dict)
                        
                        # select features from dict BTR_analysis in TAASSC that are in chosen features list 
                        syn_filt = {k: syn[k] for k in LIST_F if k in syn}
                        complexity_score.append([round(v,2) for v in syn_filt.values()])
                        utt_score.append(complexity_score) 

                if uttL[1] in ['0','3']: 
                    binary_label = 0
                else:
                    binary_label = 1
                
                clean_score = re.sub('(\[|\])', '',  str(utt_score))
                complexity.append("%s, %s"%(binary_label, clean_score)) 
                file.write("%s, %s"%(binary_label, clean_score))
                file.write("\n")

        print("Finished computing complexity")


if __name__ =="__main__":

        f = open("sentencepair_1:5.txt")
        feature_list = ['n_sent','wrd_nchr','dale_read','n_polysyl','ttr','mtld','nwords',
                        'nn_all','prep_phrase','verb','discourse_particle','contraction',
                        'cc_clause','mean_nominal_deps','np','relcl_dep','all_clauses',
                        'finite_clause','amod_nominal','det_nominal','mean_verbal_deps','mlc',
                        'dc_c','infinitive_prop','nonfinite_prop']

        lines = [l.rstrip() for l in f.readlines()]
        wr = open("F13sentencepair_1:5.txt", 'w')
        feature_extractor(lines, feature_list, wr)





