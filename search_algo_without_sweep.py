## AV-MNIST 데이터셋 활용하여 MM-NN 설계하기 위한 탐색 알고리즘(Search Algorithm)
import argparse  # 명령행 인자 처리
import random
import math
import pickle   # 객체 직렬화 및 역직렬화

from utils.wandb_utils import init_wandb
from utils.wandb_utils import log_wandb
from utils.wandb_sweep import create_sweep, run_sweep

import utils.wandb_sweep

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.config import setup
import utils.comm as comm

from backbones.ofa.model_zoo import ofa_net
from backbones.ofa.utils.layers import LinearLayer, ConvLayer

from utils.optim import RankAndCrowdingSurvival_Outer_Acc, RankAndCrowdingSurvival_Inner_Acc
from utils.save import save_ooe_population, save_results, save_resume_population

from evaluate.backbone_eval.accuracy import subnets_nas_eval
from evaluate.backbone_eval.accuracy.population_nas_eval import validate_population
from evaluate.backbone_eval.efficiency import EfficiencyEstimator
from data.data_loader import build_data_loader
from datetime import datetime
import os

# ## Sweep용 설정 파일 지정
# config_path = './config/sweep_config.yml'

# ## WandB 초기화 & 설정 불러오기
# wandb_config = init_wandb(config_path)

# # 기존 args 객체에 WandB 값 반영
# args = setup(config_path)
# for key, value in wandb_config.items():
#     setattr(args, key, value)

# print(f"Updated Config for Sweep: {args}")

directory = os.path.dirname(os.path.abspath(__name__))
directory = "../HNAS_AVMNIST"
exp_name = "tx2_avmnist"


f = directory+'/results/'+exp_name
if not os.path.exists(f):
    os.makedirs(f)

## 초기화(GPU, 난수 시드 설정) 및 설정 단계
parser = argparse.ArgumentParser(description='Harmonic-NAS Search for the Complete MM-NN Architecure')
parser.add_argument('--config-file', default=directory+'/configs/search_config_avmnist.yml')        # 설정 file로부터 하이퍼 파라미터(데이터 경로, 네트워크 설정 등) 로드
parser.add_argument('--seed', default=42, type=int, help='default random seed')
parser.add_argument('--resume-evo', default=0, type=int, help='Resume previous search')
parser.add_argument('--start-evo', default=0, type=int, help='evolution to resume')
parser.add_argument("--net", metavar="OFANET", default= "ofa_mbv3_d234_e346_k357_w1.0", help="OFA networks")


run_args = parser.parse_args()

def eval_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu          # local rank, local machine cuda id
    args.local_rank = args.gpu
    torch.cuda.set_device(args.gpu)
    print("Using GPU: ", args.gpu)

    # 랜덤 시드도 설정해 주고요
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Building  the supernets(이미지, 사운드 개별로 ㄱㄱ)
    ## 그리고, 분류기와 1st Conv. 레이어를 AV-MNIST 데이터셋에 맞게 수정한다!!
    image_supernet = ofa_net(args.net, resolution=args.image_resolution, pretrained=False, in_ch=args.in_channels, _type='avmnist')     # Once-for All 네트워크 사용 -> Supernet의 형성!!
    image_supernet.classifier = LinearLayer(image_supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    image_supernet.first_conv = ConvLayer(args.in_channels, image_supernet.first_conv.out_channels, kernel_size=image_supernet.first_conv.kernel_size,
                                 stride=image_supernet.first_conv.stride, act_func="h_swish")
    image_supernet.cuda(args.gpu)

    ## 앞에 image와 동일하게 Sound도 OFA 네트워크 설계!!
    sound_supernet = ofa_net(args.net, resolution=args.sound_resolution, pretrained=False, in_ch=args.in_channels, _type='avmnist')
    sound_supernet.classifier = LinearLayer(sound_supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    sound_supernet.first_conv = ConvLayer(args.in_channels, sound_supernet.first_conv.out_channels, kernel_size=sound_supernet.first_conv.kernel_size,
                                 stride=sound_supernet.first_conv.stride, act_func="h_swish")
    sound_supernet.cuda(args.gpu)

    # # 하드웨어 효율성 측정 위하여 LUT(Look Up Table) 불러오기 => 사실상 의미 없어서 제거 고려!!
    # lut_data_image = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)
    # lut_data_sound = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)
    
    ## load dataset, train_sampler: distributed
    train_loader, test_loader, train_sampler = build_data_loader(args)
    val_loader = None
    
    # 사전 학습된 SuperNet 가중치 로드
    assert args.pretrained_path1 and args.pretrained_path2
    image_supernet.load_weights_from_pretrained_supernet(args.pretrained_path1)
    sound_supernet.load_weights_from_pretrained_supernet(args.pretrained_path2)

    # 진화 알고리즘의 초기 단계
    if(args.resume_evo == 0): # 첫 세대부터 새로운 탐색 시작!!(진화 알고리즘 탐색)
        parent_ooe_popu = []
        parent_ooe_popu1 = image_supernet.init_population(n_samples=args.evo_search_outer.parent_popu_size)     # 이미지 모달리티 백본 8개 생성(parent_popu_size)
        parent_ooe_popu2 = sound_supernet.init_population(n_samples=args.evo_search_outer.parent_popu_size)     # 사운드 모달리티 백본 8개 생성(parent_popu_size)
        for idx in range(len(parent_ooe_popu1)):
            parent_ooe_popu1[idx]['net_id'] = f'net_{idx}_evo_0_{idx}'
            parent_ooe_popu2[idx]['net_id'] = f'net_{idx}_evo_0_{idx}'
            couple = {'backbone1': parent_ooe_popu1[idx], 'backbone2': parent_ooe_popu2[idx], 'net_id': f'net_evo_0_{idx}'}
            parent_ooe_popu.append(couple)      # 각 백본 개체의 정보를 저장한다!!
        args.start_evo = 0
        save_ooe_population(directory, 0, parent_ooe_popu, exp_name)    # 첫 번째 세대 지정하여, 이후에 이어서 학습 되게끔!

    # 저장된 population 로드 후 탐색 재개!!
    else:
        print('Resuming from population', args.start_evo)
        # 저장된 population 불러오기
        with open(directory + f'/results/{exp_name}/popu/resume_{args.start_evo}.popu', 'rb') as f:
            parent_ooe_popu = pickle.load(f)        # 이전 세대 개체 복원
        for idx in range(len(parent_ooe_popu)):
            parent_ooe_popu[idx]['net_id'] = f'net_evo_{args.start_evo}_{idx}'
        print(len(parent_ooe_popu), args.evo_search_outer.parent_popu_size)
        assert len(parent_ooe_popu) == args.evo_search_outer.parent_popu_size       # 불러온 개체 수 ==  parent_popu_size와 동일한가??

    # Run the first optimization step here --> explore backbones(진화 알고리즘의 핵심 반복)
    for evo_outer in range(args.start_evo, args.evo_search_outer.evo_iter):
        print(f"The evolution is at iteration: {evo_outer} with population size: {len(parent_ooe_popu)}")

        # Population 설정
        backbones1 = [cfg['backbone1'] for cfg in parent_ooe_popu]
        backbones2 = [cfg['backbone2'] for cfg in parent_ooe_popu]
    
        print(f"Before validation Evolution {evo_outer} Len Initial Backbones1: {len(backbones1)} Len Initial Backbones2: {len(backbones2)}") 

        ## Unimodal Backbone Eval & Select (LUT 테이블 제거로 인한, 정확도 기반 평가로 진행)
        # Evauate the backbones population on our performance metrics for the Image modality(백본 평가하기 - 이미지)
        backbones1 = validate_population(train_loader=train_loader, val_loader=test_loader, 
                        supernet=image_supernet, population=backbones1, 
                        args=args, 
                        # lut_data=lut_data_image,
                        modal_num=0, bn_calibration=True, in_channels=1, resolution=28)
    
        # Evauate the backbones population on our performance metrics for the Audio modality(백본 평가하기 - 오디오)
        backbones2 = validate_population(train_loader=train_loader, val_loader=test_loader, 
                        supernet=sound_supernet, population=backbones2, 
                        args=args, 
                        # lut_data=lut_data_sound,
                        modal_num=1, bn_calibration=True, in_channels=1, resolution=20)

        print(f"Evolution {evo_outer} Len Initial Backbones1: {len(backbones1)} Len Initial Backbones2: {len(backbones2)}")
        save_results(directory, evo_outer, 'Init_B1', backbones1, exp_name)
        save_results(directory, evo_outer, 'Init_B2', backbones2, exp_name)
        
        # Selection the promising backbones for the second stage of fusion search(평가된 백본 중 성능이 우수(정확도 기준)한 개체 선택)
        backbones1 = RankAndCrowdingSurvival_Inner_Acc(backbones1, normalize=None, 
                                                      n_survive=math.ceil(args.evo_search_outer.parent_popu_size*args.evo_search_inner.survival_ratio))

        
        backbones2 = RankAndCrowdingSurvival_Inner_Acc(backbones2, normalize=None, 
                                                      n_survive=math.ceil(args.evo_search_outer.parent_popu_size*args.evo_search_inner.survival_ratio))
        
        log_wandb(evo_outer, backbones1, backbones2, parent_ooe_popu, args)

        print(f"Evolution {evo_outer} Len Survived Backbones1: {len(backbones1)} Len Survived Backbones2: {len(backbones2)}")
        save_results(directory, evo_outer, 'Elites_B1', backbones1, exp_name)
        save_results(directory, evo_outer, 'Elites_B2', backbones2, exp_name)

        # 다음 단계 위한 융합 네트워크를 구성한다.
        parent_ooe_popu = []
        for i in range(len(backbones1)):
            id = f'net_evo_{evo_outer}_{i}'  # 개체의 고유 ID 생성
            b1 = backbones1[i]      # 이미지 모달리티 백본
            b2 = backbones2[i]      # 사운드 모달리티 백본
            b1['net_id'] = id
            b2['net_id'] = id
            
            # Exploring the fusion network macro-architecure
            steps = 2          # the number of fusions cells 
            node_steps = 1     # the number of fusion operators inside the cell 
            steps_candidates = [1, 3, 4]        # Fusion Cell 개수의 후보
            node_steps_candidates = [2, 3, 4]   # 각 Fusion Cell 내에서 수행될 Operator의 개수
            
            if(random.random() < 0.4):
                steps = random.choice(steps_candidates)  
            if(random.random() < 0.4):
                node_steps = random.choice(node_steps_candidates)
            
            
            
            couple = {'backbone1': b1, 
                      'backbone2': b2, 
                      'net_id': id, 
                      'steps': steps,           # Fusion Cell이 몇 개?
                      'node_steps': node_steps  # Fusion Operator가 몇 개?(각 Cell 별로)
                      }
            parent_ooe_popu.append(couple)
            

        print(f"Evolution {evo_outer} Len MM-Population before the fusion {len(parent_ooe_popu)}")
        print("###################################################")
        save_results(directory, evo_outer, 'Fusion_Popu', parent_ooe_popu, exp_name)

        save_ooe_population(directory, evo_outer, parent_ooe_popu, exp_name)        # 융합 네트워크 저장

        # Subnet 준비하기   
        my_subnets_to_be_evaluated = {
            cfg['net_id']: cfg for cfg in parent_ooe_popu
        }  
       
        print(f"Evolution: {evo_outer}")

        # Fusion search(백본들을 융합 탐색): Deriving the MM-NNs
        eval_results = subnets_nas_eval.avmnist_fusion_validate(
            eval_subnets1=my_subnets_to_be_evaluated,
            train_loader=train_loader,
            test_loader=test_loader,
            model1=image_supernet,
            model2=sound_supernet,
            args=args,
            bn_calibration=True
        )
        
        # Load actual population
        with open(directory + f'/results/{exp_name}/popu/evo_{evo_outer}.popu', 'rb') as f:
            actual_popu = pickle.load(f)
        assert len(actual_popu) == math.ceil(args.evo_search_outer.parent_popu_size * args.evo_search_inner.survival_ratio)


        for i, row in enumerate(actual_popu):
            mm_id = actual_popu[i]['net_id']

            b1_id = str(eval_results[i]['net_id1'])
            b2_id = str(eval_results[i]['net_id2'])

            steps = int(eval_results[i]['steps'])
            node_steps = int(eval_results[i]['node_steps'])
            genotype = eval_results[i]['genotype']

            # Update population with fusion results
            for mm in actual_popu:
                if mm['backbone1']['net_id'] == b1_id:
                    b1 = mm['backbone1'].copy()
                    break
                
            for mm in actual_popu:
                if mm['backbone2']['net_id'] == b2_id:
                    b2 = mm['backbone2'].copy()
                    break

            b1['net_id'] = mm_id
            b2['net_id'] = mm_id

            actual_popu[i].update({
                'Acc@1': eval_results[i]['Acc@1'],
                'latency': eval_results[i]['latency'],
                'energy': eval_results[i]['energy'],
                'backbone1': b1,
                'backbone2': b2,
                'steps': steps,
                'node_steps': node_steps,
                'genotype': genotype
            })

        print(f"Evolution {evo_outer} Len Fusion Results {len(eval_results)}")
        print("###################################################")
        save_results(directory, evo_outer, 'Fusion_Popu_Results', actual_popu, exp_name)

        # Survive the best models(상위 융합 네트워크 선택 -> 다음 세대로 넘기기)
        n_survive = math.ceil(
            math.ceil(args.evo_search_outer.parent_popu_size * args.evo_search_inner.survival_ratio) * args.evo_search_outer.survival_ratio
        )
        survivals_ooe = RankAndCrowdingSurvival_Outer_Acc(pop=actual_popu, normalize=None, n_survive=n_survive)

        print(f"Evolution {evo_outer} Len Fusion Survivals {len(survivals_ooe)}")
        print("###################################################")
        save_results(directory, evo_outer, 'Elites_MM', survivals_ooe, exp_name)

            
        # Generate the next population of the backbones for the next evolution(교차, 변이를 통한 새로운 세대 생성)
        parent_ooe_popu = []

        # crossover  (this removes the net_id key)
        for idx in range(args.evo_search_outer.crossover_size):
            if len(parent_ooe_popu) >= args.evo_search_outer.parent_popu_size:
                break  # 초과 방지
            ## 상위 개체에서 2개의 부모 선택!!
            cfg1 = random.choice(survivals_ooe)     
            cfg2 = random.choice(survivals_ooe)
            # print("config 1",cfg1)

            ## 부모 개체의 백본 네트워크 정보 교차!!
            cfg_backbone1 = image_supernet.crossover_and_reset1(cfg1['backbone1'], 
                                                                       cfg2['backbone1'], 
                                                                       crx_prob=args.evo_search_outer.crossover_prob)
            cfg_backbone2 = image_supernet.crossover_and_reset1(cfg1['backbone2'], 
                                                                       cfg2['backbone2'], 
                                                                       crx_prob=args.evo_search_outer.crossover_prob)
            cfg = {'backbone1': cfg_backbone1, 'backbone2': cfg_backbone2}      # 새로운 백본 NETWORK 생성!!
            parent_ooe_popu.append(cfg)

        # Mutation  (변이 알고리즘)
        for idx in range(args.evo_search_outer.mutate_size):    
            if len(parent_ooe_popu) >= args.evo_search_outer.parent_popu_size:
                break  # 초과 방지      
            old_cfg = random.choice(survivals_ooe)
            
            cfg_backbone1 = image_supernet.mutate_and_reset(old_cfg['backbone1'], prob=args.evo_search_outer.mutate_prob)
            cfg_backbone2 = image_supernet.mutate_and_reset(old_cfg['backbone2'], prob=args.evo_search_outer.mutate_prob)
            cfg = {'backbone1': cfg_backbone1, 'backbone2': cfg_backbone2}
            parent_ooe_popu.append(cfg)

        # Population ID 수정
        print("len parent_ooe_popu: {} / the correct: {}".format(len(parent_ooe_popu), args.evo_search_outer.parent_popu_size))
        assert len(parent_ooe_popu) == args.evo_search_outer.parent_popu_size

        for idx in range(len(parent_ooe_popu)):
            parent_ooe_popu[idx]['net_id'] = f'net_evo_{evo_outer}_{idx}'  # GPU 관련 부분 제거
            parent_ooe_popu[idx]['backbone1']['net_id'] = f'net_evo_{evo_outer}_{idx}'
            parent_ooe_popu[idx]['backbone2']['net_id'] = f'net_evo_{evo_outer}_{idx}'
            
        print("Generation {} the parent popu is {}: ".format(evo_outer, len(parent_ooe_popu)))
        
        log_wandb(evo_outer, backbones1, backbones2, survivals_ooe, args)

        # Population 저장
        save_resume_population(directory, evo_outer + 1, parent_ooe_popu, exp_name)
        print("Evolution {} has finished {}:".format(evo_outer, datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))
    
    print("The search has ended at :",datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))


        
if __name__ == '__main__':
    # Sweep용 설정 파일 지정
    project_name = 'search_algo_harmnas'
    entity='junylee00-chonnam-national-university'
    config_path = './configs/search_config_avmnist.yml' 

    # WandB 초기화 및 설정 불러오기
    wandb_config = init_wandb(project_name, config_path)

    args = setup(run_args.config_file)      # args 객체 내에 하이퍼 파라미터, 데이터 경로, 네트워크 설정 등 불러옴
    args.resume_evo = run_args.resume_evo
    args.start_evo = run_args.start_evo
    args.net = run_args.net


    # GPU 설정
    if torch.cuda.is_available():
        args.gpu = 0  # 단일 GPU 사용
        torch.cuda.set_device(args.gpu)
        print("Using GPU: ", args.gpu)
    else:
        print("CUDA is not available. Exiting...")
        exit(1)

    print("The search has started at :", datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    # 단일 GPU로 eval_worker 실행
    eval_worker(args.gpu, 1, args)  # ngpus_per_node를 1로 설정
