import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import utils.comm as comm

from evaluate.backbone_eval.efficiency import look_up_ofa_proxy
from .single_subnet_eval import validate_one_subnet

from fusion_search.train_fusion_search import train_darts_model
from fusion_search.search.darts.utils import count_genotype_hardware_metrics

import random 


def validate(
    subnets_to_be_evaluated,
    train_loader, 
    val_loader,
    test_loader, 
    model, 
    lut_data,
    args, 
    bn_calibration=False,
):
    supernet = model.module \
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    results = []
    with torch.no_grad():
        for net_id in subnets_to_be_evaluated:
            if net_id == 'supernet_nas_min_net': 
                supernet.set_min_net()
            elif net_id == 'supernet_nas_max_net':
                supernet.set_max_net()
            elif net_id.startswith('supernet_nas_random_net'):
                supernet.sample_active_subnet()
            else:
                supernet.set_active_subnet(
                    subnets_to_be_evaluated[net_id]['ks'],
                    subnets_to_be_evaluated[net_id]['e'],
                    subnets_to_be_evaluated[net_id]['d'],
                )

            subnet = supernet.get_active_subnet()
            
            subnet.cuda(args.gpu)

            if bn_calibration:
                subnet.eval()
                subnet.reset_running_stats_for_calibration()

                # estimate running mean and running statistics
                for batch_idx, (input_data, target) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break

                    input_data = input_data.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
                    
                    subnet(input_data)  #forward only

            f1w, f1m = validate_one_subnet(val_loader, subnet, args) 
            lat, enrg = look_up_ofa_proxy(net=subnet, lut=lut_data, supernet=args.supernet_arch)
            print("subnet f1:" , f1w)
            summary = str({
                        'net_id': net_id,
                        'mode': 'evaluate',
                        'epoch': getattr(args, 'curr_epoch', -1),
                        'F1-W@1': 100-f1w,
                        'latency': lat,
                        'energy': enrg,
            })

            if args.distributed and getattr(args, 'distributed_val', True):
                results += [summary]
            else:
                group = comm.reduce_eval_results(summary, args.gpu)
                results += group

    return results

def avmnist_fusion_validate(
    eval_subnets1,
    train_loader, 
    test_loader, 
    model1,
    model2, 
    args, 
    bn_calibration=False,
):
    ## 서브넷(백본) 설정
    supernet1 = model1.module \
        if isinstance(model1, torch.nn.parallel.DistributedDataParallel) else model1

    supernet2 = model2.module \
        if isinstance(model2, torch.nn.parallel.DistributedDataParallel) else model2

    results = []
    
    # print('gpu {} len(eval_subnets1): {} / len(eval_subnets2): {} '.format(args.gpu, len(eval_subnets1), len(eval_subnets2)))
    
    # why torch no grad?
    with torch.no_grad():
        
        print("inside fusion validate eval_subnets1:  {} len {}".format(eval_subnets1,len(eval_subnets1)))
        id_s = list(eval_subnets1.keys())
        # print("the keeeeeeeeeys are : ", id_s)

        ## 평가할 서브넷 리스트를 랜덤히 섞기
        random.shuffle(id_s)
        idx1_list = id_s.copy()
        random.shuffle(id_s)
        idx2_list = id_s.copy()
        print("gpu {} list1 {} list2 {}".format(args.gpu, idx1_list, idx2_list))
        
        for i in range(len(eval_subnets1)):         # 앞 단계에서 걸러진 2개의 백본 가지고 평가 ㄱㄱ
            idx1 = idx1_list[i]
            idx2 = idx2_list[i]
            net_id1 = eval_subnets1[idx1]['backbone1']['net_id']
            net_id2 = eval_subnets1[idx2]['backbone2']['net_id']
            
            assert idx1 == eval_subnets1[idx1]['backbone1']['net_id']
            assert idx2 == eval_subnets1[idx2]['backbone2']['net_id']
            # net_id1 = choice(list(eval_subnets1.keys()))
            # net_id2 = choice(list(eval_subnets2.keys()))
            
            ## 선택된 백본의 활성화!!
            if(net_id1 == 'supernet_nas_min_net'):
                supernet1.set_min_net()
            elif net_id1 == 'supernet_nas_max_net':
                supernet1.set_max_net()
            elif net_id1.startswith('supernet_nas_random_net'):
                supernet1.sample_active_subnet()
            else:
                supernet1.set_active_subnet(
                    eval_subnets1[net_id1]['backbone1']['ks'],
                    eval_subnets1[net_id1]['backbone1']['e'],
                    eval_subnets1[net_id1]['backbone1']['d'],
                )
        
            if(net_id2 == 'supernet_nas_min_net'):
                supernet2.set_min_net()
            elif net_id2 == 'supernet_nas_max_net':
                supernet2.set_max_net()
            elif net_id2.startswith('supernet_nas_random_net'):
                supernet2.sample_active_subnet()
            else:
                supernet2.set_active_subnet(
                    eval_subnets1[net_id2]['backbone2']['ks'],
                    eval_subnets1[net_id2]['backbone2']['e'],
                    eval_subnets1[net_id2]['backbone2']['d'],
                )

            ## 융합 네트워크의 생성
            steps = eval_subnets1[idx1]['steps']
            node_steps = eval_subnets1[idx1]['node_steps']
            subnet1 = supernet1.get_active_subnet()
            subnet2 = supernet2.get_active_subnet()
            subnet1.cuda(args.gpu)
            subnet2.cuda(args.gpu)

            ## BN Calibration 수행 => BatchNorm 통계 재학습 ㄱㄱ
            if bn_calibration:
                subnet1.eval()
                subnet1.reset_running_stats_for_calibration()
                subnet2.eval()
                subnet2.reset_running_stats_for_calibration()
            
                # estimate running mean and running statistics
                for batch_idx, (input_data1, input_data2, target) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break

                    input_data1 = input_data1.cuda(args.gpu, non_blocking=True)
                    input_data2 = input_data2.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)            
                    subnet1(input_data1)  #forward only
                    # print("subnet2:", subnet2)
                    subnet2(input_data2)  #forward only
                
                
            # At this stage we have sampled two random subnets(2개의 서브넷 랜덤으로 ㄱㄱ)
            print("gpu {}  id1: {} / id2: {}".format(args.gpu, net_id1, net_id2))


            # fusion search for one subnet(train, test 데이터로더 같이 딕셔너리 형태 선언)
            dataloaders = {
                    'train': train_loader,
                    'test': test_loader
                    }
            
            
            ## 서브넷의 출력 채널 수??
            # 1. 이미지 슈퍼넷
            imagenet = subnet1
            subnet1_channels = []
            out = imagenet(torch.randn(2, 1, 28, 28).cuda(args.gpu))
            for i in range(len(out)):
                subnet1_channels.append(out[i].shape[1])
            
            chosen_channels_idx1 = []
            num_chosen_blocks1 = 4
            
            offset = (len(subnet1_channels)-1) // num_chosen_blocks1
            idx = len(subnet1_channels)-2
            chosen_channels_idx1.append(idx)
            for i in range(num_chosen_blocks1-1):
                ## 전체 블록을 균등한 간격(offset)으로 나눠 선택 ㄱㄱ
                idx -= offset
                chosen_channels_idx1.append(idx)
            chosen_channels_idx1.reverse()

            # 2. 사운드 슈퍼넷
            soundnet = subnet2
            subnet2_channels = []
            out = soundnet(torch.randn(2, 1, 20, 20).cuda(args.gpu))
            for i in range(len(out)):
                subnet2_channels.append(out[i].shape[1])
            ## 사운드 모달리티의 선택된 블록 인덱스~
            chosen_channels_idx2 = []
            num_chosen_blocks2 = 4
            offset = (len(subnet2_channels)-1) // num_chosen_blocks2
            idx = len(subnet2_channels)-2
            chosen_channels_idx2.append(idx)
            for i in range(num_chosen_blocks2-1):
                ## 전체 블록을 균등한 간격(offset)으로 나눠 선택 ㄱㄱ
                idx -= offset
                chosen_channels_idx2.append(idx)
            chosen_channels_idx2.reverse()


            args.num_input_nodes  = num_chosen_blocks1 + num_chosen_blocks2

        


            # Run the fusion search            
            fusion_acc, fusion_genotype = train_darts_model(dataloaders=dataloaders,args=args, gpu=args.gpu, 
                        network1=imagenet, 
                        network2=soundnet, 
                        chosen_channels_idx1=chosen_channels_idx1, 
                        chosen_channels_idx2=chosen_channels_idx2,
                        subnet1_channels=subnet1_channels, 
                        subnet2_channels=subnet2_channels, 
                        phases=['train', 'test'], 
                        steps=steps, 
                        node_steps=node_steps)
        
            
            
            fusion_metrics = count_genotype_hardware_metrics(fusion_genotype, args)
            

            
            print("Steps: {}  Node_Steps: {} Genotype: {}".format(steps, node_steps, fusion_genotype))
            acc1 = eval_subnets1[net_id1]['backbone1']['Acc@1']
            acc2 = eval_subnets1[net_id2]['backbone2']['Acc@1']
            
            print("Subnet1 {:.2f} Subnet2 {:.2f} Fusion {:.2f} ".format(acc1, acc2, fusion_acc))
            
            ## F1-score 기반 평가일 때 사용!!(in 멀티레이블 분류 문제)
            # subnet_f1w, f1m = validate_one_subnet(test_loader, subnet1, args) 
            # confirm_acc  = new_validate_one_subnet(test_loader, subnet1, args) 
            # confirm_nodist_f1  = nodist_validate_one_subnet(test_loader, subnet1, args) 
            # print("f1 subnet: ", subnet_f1w)
            
            # lat1, enrg1 = look_up_ofa_proxy(net=subnet1, lut=lut_data1, resolution=28, supernet=args.supernet_arch, num_channels=1)
            # lat2, enrg2 = look_up_ofa_proxy(net=subnet2, lut=lut_data2, resolution=20, supernet=args.supernet_arch, num_channels=1)
            lat1 = eval_subnets1[net_id1]['backbone1']['latency']
            lat2 = eval_subnets1[net_id2]['backbone2']['latency']
            enrg1 = eval_subnets1[net_id1]['backbone1']['energy']
            enrg2 = eval_subnets1[net_id2]['backbone2']['energy']
            # acc1 = new_validate_one_subnet(test_loader, subnet1, args, modal_num=0)
            # acc2 = new_validate_one_subnet(test_loader, subnet2, args, modal_num=1)

            ## 디버깅용 정확도, latency, energy 비교
            # print("apres la fusion {} replace the acc1 by {} / {}  ".format(acc1==eval_subnets1[net_id1]['backbone1']['Acc@1'], acc1, eval_subnets1[net_id1]['backbone1']['Acc@1'] ))
            # print("apres la fusion {} replace the acc2 by {} / {} ".format(acc2==eval_subnets1[net_id2]['backbone2']['Acc@1'], acc2, eval_subnets1[net_id2]['backbone2']['Acc@1'] ))
            # print("{} replace lat1 by {} / {} ".format(lat1==eval_subnets1[net_id1]['backbone1']['latency'], lat1, eval_subnets1[net_id1]['backbone1']['latency']))
            # print("{} replace lat2 by {} / {} ".format(lat2==eval_subnets1[net_id2]['backbone2']['latency'], lat2, eval_subnets1[net_id2]['backbone2']['latency']))
            # print("{} replace enrg1 by {} / {} ".format(enrg1==eval_subnets1[net_id1]['backbone1']['energy'], enrg1, eval_subnets1[net_id1]['backbone1']['energy']))
            # print("{} replace enrg2 by {} / {}".format(enrg2==eval_subnets1[net_id2]['backbone2']['energy'], enrg2, eval_subnets1[net_id2]['backbone2']['energy']))
            lat = lat1 + lat2 
            enrg = enrg1 + enrg2
            
            summary = str({

                        'net_id1': net_id1,
                        'net_id2': net_id2,
                        'steps': steps,
                        'node_steps': node_steps,
                        'mode': 'evaluate',
                        'epoch': getattr(args, 'curr_epoch', -1),
                        'Acc@1': fusion_acc,
                        'latency': lat + fusion_metrics['lat'],
                        'energy': enrg + fusion_metrics['enrg'],
                        'genotype': str(fusion_genotype),
            })

            if args.distributed and getattr(args, 'distributed_val', True):
                results += [summary]
            else:
                group = comm.reduce_eval_results(summary, args.gpu)
                results += group
    print("Fin fusion GPU {} len(results): {}".format(args.gpu, len(results)))
    return results