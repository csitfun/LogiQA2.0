import json
import time
import openai
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
openai.api_key = '' 

incontext = "Given the fact: All Cantonese are southerners. Some Cantonese don't like chili. Does it follow that: Some southerners don't like chili. Yes or no? yes\nGiven the fact: It is difficult for cactus to survive in humid climates; citrus is difficult to grow in cold climates. In most parts of a province, at least one species is not difficult to survive and grow between cactus and citrus. Does it follow that: Half of the province is humid and cold. Yes or no? no\nGiven the fact: It is difficult for cactus to survive in humid climates; citrus is difficult to grow in cold climates. In most parts of a province, at least one species is not difficult to survive and grow between cactus and citrus. Does it follow that: Most of the province is hot. Yes or no? no\nGiven the fact: It is difficult for cactus to survive in humid climates; citrus is difficult to grow in cold climates. In most parts of a province, at least one species is not difficult to survive and grow between cactus and citrus. Does it follow that: Most of the province is either dry or warm. Yes or no? yes\n"
def gpt3_api(prompt):
   response = openai.Completion.create(
      model="text-davinci-002",
      prompt=incontext + prompt,
      temperature=0,
      max_tokens=60,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
   )
   return response

with open('test1.txt') as f:
   c = 0
   y_true = []
   y_pred = []
   lines = f.readlines()
   for i, line in enumerate(lines):
      line_dict = json.loads(line)

      label = 0 if line_dict['label']=="not entailed" else 1
      maj_premise = ' '.join(line_dict['major_premise'])
      min_premise = ' '.join(line_dict['minor_premise'])
      hypo = line_dict['conclusion']
      prompt_input = "Given the fact: " + maj_premise + ' ' + min_premise + " Does it follow that: " + hypo + " Yes or no?"

      y_true.append(label)
      prompt = prompt_input
      output = gpt3_api(prompt)
      time.sleep(5)
      pred = output.choices[0].text.lower()
      y_pred.append(pred)

   print(y_true)
   print(y_pred)
   f_score = f1_score(y_true, y_pred, average='binary')
   p_score = precision_score(y_true, y_pred, average='binary')
   r_score = recall_score(y_true, y_pred, average='binary')
   acc = accuracy_score(y_true, y_pred)
   print(f_score)
   print(p_score)
   print(r_score)
   print(acc)
