import json
import pickle
import os

## 특정 세대의 population 저장 
def save_ooe_population(dir, evo, popu, dataset):
    folder = dir+'/results/' + dataset + '/popu'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(dir+'/results/' + dataset + '/popu/evo_'+str(evo)+'.popu', 'wb') as f: 
        pickle.dump(popu, f)

## 진화 알고리즘의 결과 저장!!
def save_results(dir, evo, name, popu, dataset):

    folder = dir+'/results/'+dataset+'/ooe/evo_'+str(evo)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    with open(folder+'/evo_'+str(evo)+'_'+name+'.json', 'w') as f:
        json.dump(popu, f)

## 현재까지 진행된 탐색을 저장 -> 이후 다시 이어서 실행!!
def save_resume_population(dir, evo, popu, dataset):
    folder = dir+'/results/' + dataset + '/popu'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(dir+'/results/' + dataset + '/popu/resume_'+str(evo)+'.popu', 'wb') as f: 
        pickle.dump(popu, f)





# def save_results_ioe(dir, evo, popu):
#     with open(dir+'/results/ioe/evo_'+str(evo)+'_eval.json', 'w') as f:
#         json.dump(popu, f)

# def save_results_ooe(dir, evo, popu):
#     with open(dir+'/results/ooe/evo_'+str(evo)+'_eval.json', 'w') as f:
#         json.dump(popu, f)
        
        
