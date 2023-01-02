import json
import pandas as pd
from itertools import *

#preprocessing json files

def convert_json(f, outfile, min_length_dialogue = 3, max_length_dialogue=61):
  """converts json dataset into a csv: turn_id ; left_context ; label(0/1) """
  data = json.load(f)
  j_wr = open(outfile, "w")

  utterances = []
  convergence = []
  turn = []
  dialog_id = []

  for d in data:
    if len(d["turns"])> min_length_dialogue and len(d["turns"]) < max_length_dialogue:
      for values in d['turns']:
          dialog_id.append(d['dialogue_id'])
          turn.append(values['turn_id'])
          utterances.append(values['utterance'])
          if 'friction' in values.keys():
              if "incomprehension" in values["friction"]:
                  convergence.append(1)
              else:
                  convergence.append(3)
          else: 
              convergence.append(0)
                    
  dict = {"dialog_id": dialog_id, "turn_id": turn, "utterance": utterances, "convergence": convergence}
  df = pd.DataFrame.from_dict(dict)  
  
  # iterate over columns
  for i in range(len(df['utterance'])):
      new_file = (
      str(df['dialog_id'][i]) + "\t" + str(
          df['turn_id'][i]) + "\t" +str(
              df['utterance'][i])+ "\t" + str(
                  df['convergence'][i])+'\n')
      j_wr.write(new_file)
 

if __name__ =="__main__":
  f = "data/frictionTM.json"
  wr = "data/frictionTM.txt"

  convert_json(f, wr)
