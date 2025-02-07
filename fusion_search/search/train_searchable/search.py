import torch
import fusion_search.auxiliary.scheduler as sc
import copy
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import os
from IPython import embed
from fusion_search.search.darts.utils import count_parameters, save, save_pickle, count_supernet_hardware_metrics
import torch.nn.functional as F

import utils.wandb_utils as wu

## DARTS 기반의 탐색 수행!!
def train_mmimdb_track_f1(model, architect,
                        criterion, optimizer, scheduler, dataloaders,
                        dataset_sizes, device, num_epochs, 
                        parallel, logger, plotter, args,
                        init_f1=0.0, th_fscore=0.3, 
                        status='search', save_logger=False, save_model=False, plot_arch=False, phases=['train', 'dev', 'test'], steps=0, node_steps=0):

    f1_type = "macro"  # 일단 초기화
    if(args.task=='multilabel'):
        f1_type=args.f1_type        # 멀티라벨 task => 사용자 정의 F1-score 사용!!

    ## 탐색 과정에서 최고의 성능을 기록하기 위한 변수들(genotype, F1-score)
    best_genotype = None
    best_f1 = init_f1
    best_epoch = 0

    best_test_genotype = None
    best_test_f1 = init_f1
    best_test_epoch = 0

    ## 학습 loop 시작!!
    failsafe = True
    cont_overloop = 0
    while failsafe:     # 예외 처리용 Flag => 정상적으로 훈련 마무리 안 되면 한번 더 ㄱㄱ
        for epoch in range(num_epochs):    
            print(f"{epoch}번째 Epoch 시작!!")
            
            if(save_logger):
                logger.info('Epoch: {}'.format(epoch))            
                logger.info("EXP: {}".format(args.save) )

            # phases = []
            # if status == 'search':
            #     phases = ['train', 'dev', 'test']
            # else:
            #     # while evaluating, add dev set to train also
            #     phases = ['train', 'dev', 'test']
            
            # Each epoch has a training and validation phase
            
            
            print("Inside train_track phases: ", phases)
            for phase in phases:
                ## train=> DARTS 가중치 학습 & Fusion Network 가중치 업데이트
                if phase == 'train':
                    if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                        scheduler.step()
                    if architect is not None:

                        if(save_logger):
                            architect.log_learning_rate(logger)
                    model.train()  # Set model to training mode
                    list_preds = [] 
                    list_label = []   
                ## dev => DARTS 활용하여 Fusion Network 구조 학습!!            
                elif phase == 'dev':
                    if status == 'eval':
                        if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.step()
                    model.train()       # 탐색 과정에서도 모델 학습 ㄱㄱ
                    list_preds = [] 
                    list_label = [] 
                ## test => 최종 평가!!             
                else:
                    # model.eval()  # Set model to evaluate mode
                    list_preds = [] 
                    list_label = []                    
    
                
                running_loss = 0.0
                running_f1 = init_f1

                with tqdm(dataloaders[phase]) as t:
                    # Iterate over data.
                    ## 데이터 로딩 및 모델 학습하기
                    for data in dataloaders[phase]:
                        modality1, modality2, label = data[0], data[1], data[-1]
        
                        # device
                        modality1 = modality1.to(device)
                        modality2 = modality2.to(device)                
                        label = label.to(device)

                        # GPU로 데이터 전송
                        if status == 'search' and (phase == 'dev' or phase == 'test'):
                            with torch.set_grad_enabled(True):
                                architect.step((modality1, modality2), label, logger, args)
                            
                        # Optimizer 일단 초기화!!
                        optimizer.zero_grad()
        
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train' or (phase == 'dev' and status == 'eval')):
                            ## Forward & Loss 계산하기!!
                            output = model((modality1, modality2))
                            
                            if (isinstance(output, tuple) or isinstance(output, list)):
                                output = output[-1]
        
                            _, preds = torch.max(output, 1)     # 그리고 preds는 최종 예측값!!
                            
                            ## LUT 기반의 하드웨어 성능 평가
                            # hardware_evaluation = count_supernet_hardware_metrics(model, args, steps, node_steps)
                            if args.fusion_lut is not None:     # ✅ LUT가 있을 때만 실행!
                                hardware_evaluation = count_supernet_hardware_metrics(model, args, steps, node_steps)
                            else:
                                print("⚠️ Warning: Skipping hardware evaluation since no Fusion LUT is provided.")
                                hardware_evaluation = {"lat": 0.0, "enrg": 0.0}  # 기본값 반환

                            
                            ## 기존 손실 함수에 하드웨어 성능 고려!!
                            a = 0.5
                            b = 0.5
                            c = 0.5
                            loss = (criterion(output, label)**a) + (hardware_evaluation['lat']**b) + (hardware_evaluation['enrg']**c)
                            # loss = criterion(output, label)
                        
                            if(args.task=='multilabel'):
                                preds_th = torch.sigmoid(output) > th_fscore
                            elif(args.task=='classification'):
                                preds_th = output
                            else:
                                raise NotImplementedError    
                                
                            
                            ## 모델 학습 및 최적의 구조 선택!!
                            # backward + optimize only if in training phase
                            if phase == 'train' or (phase == 'dev' and status == 'eval'):
                                if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                    scheduler.step()
                                    scheduler.update_optimizer(optimizer)
                                loss.backward()
                                optimizer.step()
                            
                            # if phase == 'dev':
                            list_preds.append(preds_th.cpu())
                            list_label.append(label.cpu()) 

                        # statistics
                        running_loss += loss.item() * modality1.size(0)     # 1 epoch간 손실 누적!!

                        batch_pred_th = preds_th.data.cpu() # 모델이 예측한 값
                        batch_true = label.data.cpu()       # 실제 정답 라벨

                        # if(args.task=='classification'):
                            
                        ## 평가 지표) multilabel => F1-score, Classification => accuracy score    
                        if(args.task=='multilabel'):
                            batch_f1 = f1_score(batch_pred_th.numpy(), batch_true.numpy(), average=f1_type, zero_division=1)                  
                        elif(args.task=='classification'):
                            batch_pred_th = torch.argmax(batch_pred_th, 1)
                            batch_f1 = accuracy_score(batch_pred_th.numpy(), batch_true.numpy())

                        ## 진행 상황의 batch 단위 출력!!
                        if(args.task=='multilabel'):
                            postfix_str = 'batch_loss: {:.03f}, batch_f1: {:.03f}'.format(loss.item(), batch_f1*100)
                        elif(args.task=='classification'):
                            postfix_str = 'batch_loss: {:.03f}, batch_acc: {:.03f}'.format(loss.item(), batch_f1*100)
                        
                        t.set_postfix_str(postfix_str)
                        t.update()
                            
                epoch_loss = running_loss / dataset_sizes[phase]    # 1 epoch 동안 평균 손실
                
                ## 모든 배치에서의 예측 + 실제 정답 합침
                y_pred = torch.cat(list_preds, dim=0)
                y_true = torch.cat(list_label, dim=0)
                                
                # epoch_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)  
                # if(args.task=='multilabel'):
                #     epoch_f1 = f1_score(y_true.numpy(), y_pred.numpy(), average=f1_type, zero_division=1)
                #     logger.info('{} Loss: {:.4f}, {} F1: {:.4f}'.format(phase, epoch_loss, f1_type, epoch_f1))
                # elif(args.task=='classification'):
                #     y_pred = torch.argmax(y_pred, 1)
                #     epoch_f1 = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
                #     logger.info('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_f1))
                
                ## 최종 성능 평가!!
                if(args.task=='multilabel'):
                    epoch_f1 = f1_score(y_true.numpy(), y_pred.numpy(), average=f1_type, zero_division=1)                  
                elif(args.task=='classification'):
                    y_pred = torch.argmax(y_pred, 1)
                    epoch_f1 = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
                
                print("epoch f1 is :",epoch_f1)

                wu.log_wandb_fusion(epoch, phase, epoch_loss, epoch_f1, args)

                ## 로그 저장
                if(save_logger):
                    # if(args.task=='multilabel'):
                    #     epoch_f1 = f1_score(y_true.numpy(), y_pred.numpy(), average=f1_type, zero_division=1)
                    #     logger.info('{} Loss: {:.4f}, {} F1: {:.4f}'.format(phase, epoch_loss, f1_type, epoch_f1))
                    # elif(args.task=='classification'):
                    #     y_pred = torch.argmax(y_pred, 1)
                    #     epoch_f1 = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
                    #     logger.info('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_f1))

                    if(args.task=='multilabel'):
                        logger.info('{} Loss: {:.4f}, {} F1: {:.4f}'.format(phase, epoch_loss, f1_type, epoch_f1))
                    elif(args.task=='classification'):
                        logger.info('{} Loss: {:.4f}, {} Acc: {:.4f}'.format(phase, epoch_loss, f1_type, epoch_f1))
                
                if(False):
                    pass
                # if parallel and torch.cuda.device_count() > 1:
                #     num_params = 0
                #     for reshape_layer in model.module.reshape_layers:
                #         num_params += count_parameters(reshape_layer)

                #     num_params += count_parameters(model.module.fusion_net)
                #     if(save_logger):
                #         logger.info("Fusion Model Params: {}".format(num_params) )

                #     genotype = model.module.genotype()
                else:
                    ## 모델의 Genotype(탐색된 아키텍쳐) 저장
                    num_params = 0
                    for reshape_layer in model.reshape_layers:
                        ## 모델 내 fusion network랑 reshape layer의 총 파라미터 수
                        num_params += count_parameters(reshape_layer)

                    num_params += count_parameters(model.fusion_net)
                    if(save_logger):
                        logger.info("Fusion Model Params: {}".format(num_params) )
                    # 현재 Fusion의 최적화된 아키텍처 저장 ㄱㄱ
                    genotype = model.genotype()
                if(save_logger):
                    logger.info(str(genotype))
                
                                
                ## NaN 발생 감지 ㄱ
                if phase == 'train' and epoch_loss != epoch_loss:
                    if(save_logger):
                        logger.info("Nan loss during training, escaping")
                    model.eval()              
                    return best_f1
                
                ## 최적의 네트워크 구조를 저장!!
                if phase == 'dev' and status == 'search':
                    if epoch_f1 > best_f1:
                        best_f1 = epoch_f1
                        best_genotype = copy.deepcopy(genotype)
                        best_epoch = epoch
                        
                        if(save_model):
                            if parallel:
                                save(model.module, os.path.join(args.save, 'best', 'best_model.pt'))
                            else:   # 단일 GPU 일 때
                                save(model, os.path.join(args.save, 'best', 'best_model.pt'))

                            best_genotype_path = os.path.join(args.save, 'best', 'best_genotype.pkl')
                            save_pickle(best_genotype, best_genotype_path)
                
                if phase == 'dev':
                    if epoch_f1 > best_f1:
                        print("Updating the best_dev_f1")
                        best_epoch = epoch
                        best_f1 = epoch_f1

                ## 최종 테스트에서는 최고 성능을 기록 ㄱㄱ!!
                if phase == 'test':
                    if epoch_f1 > best_test_f1:
                        print("Updating the best_test_f1")
                        best_test_f1 = epoch_f1
                        best_test_genotype = copy.deepcopy(genotype)
                        best_test_epoch = epoch
                    
                        if(save_model):
                            if parallel:
                                save(model.module, os.path.join(args.save, 'best', 'best_test_model.pt'))
                            else:
                                save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))

                            best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
                            save_pickle(best_test_genotype, best_test_genotype_path)
                            
                            print("Best saved")

            if(plot_arch):
                file_name = "epoch_{}".format(epoch)
                file_name = os.path.join(args.save, "architectures", file_name)
                plotter.plot(genotype, file_name, task=args.dataset)

            if(save_logger):
                logger.info("Current best dev {} F1: {}, at training epoch: {}".format(f1_type, best_f1, best_epoch) )
                logger.info("Current best test {} F1: {}, at training epoch: {}".format(f1_type, best_test_f1, best_test_epoch) )

        if best_f1 != best_f1 and num_epochs == 1 and cont_overloop < 1:
            failsafe = True
            if(save_logger):
                logger.info('Recording a NaN F1, training for one more epoch.')
        else:
            failsafe = False
            
        cont_overloop += 1
    
    if best_f1 != best_f1:
        best_f1 = 0.0

    # if status == 'search':
    #     return best_f1, best_genotype
    # else:
    #     return best_test_f1, best_test_genotype
    return best_test_f1, best_test_genotype





# def test_mmimdb_track_f1(  model, criterion, dataloaders,
#                            device, 
#                            parallel, logger, args,
#                            f1_type='weighted', init_f1=0.0, th_fscore=0.3):

#     best_test_genotype = None
#     best_test_f1 = init_f1
#     best_test_epoch = 0
    
#     dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['test']}
    
#     model.eval()  # Set model to evaluate mode
#     list_preds = [] 
#     list_label = []                    

#     running_loss = 0.0
#     running_f1 = init_f1
#     phase = 'test'

#     with tqdm(dataloaders[phase]) as t:
#         # Iterate over data.
#         for data in dataloaders[phase]:
#             # get the inputs
#             modality1, modality2, label = data['modality1'], data['modality2'], data['label']
#             # device
#             # print("modality1 : ",modality1.shape)
#             # print("type : ",type(modality1))
#             modality1 = modality1.to(device)
#             modality2 = modality2.to(device)                
#             label = label.to(device)

#             output = model((modality2, modality1))        
#             if (isinstance(output, tuple) or isinstance(output, list)):
#                 output = output[-1]

#             _, preds = torch.max(output, 1)
#             loss = criterion(output, label)
#             preds_th = torch.sigmoid(output) > th_fscore
#             # if phase == 'dev':
#             list_preds.append(preds_th.cpu())
#             list_label.append(label.cpu()) 

#             # statistics
#             running_loss += loss.item() * modality1.size(0)

#             batch_pred_th = preds_th.data.cpu().numpy()
#             batch_true = label.data.cpu().numpy()
#             batch_f1 = f1_score(batch_pred_th, batch_true, average='samples')  

#             postfix_str = 'batch_loss: {:.03f}, batch_f1: {:.03f}'.format(loss.item(), batch_f1)
#             t.set_postfix_str(postfix_str)
#             t.update()
                
#     epoch_loss = running_loss / dataset_sizes[phase]
    
#     # if phase == 'dev':
#     y_pred = torch.cat(list_preds, dim=0).numpy()
#     y_true = torch.cat(list_label, dim=0).numpy()

#     epoch_f1 = f1_score(y_true, y_pred, average=f1_type)                  

#     logger.info('{} Loss: {:.4f}, {} F1: {:.4f}'.format(
#                     phase, epoch_loss, f1_type, epoch_f1))
    
#     if parallel:
#         num_params = 0
#         for reshape_layer in model.module.reshape_layers:
#             num_params += count_parameters(reshape_layer)

#         num_params += count_parameters(model.module.fusion_net)
#         logger.info("Fusion Model Params: {}".format(num_params) )
#         genotype = model.module.genotype()
#     else:
#         num_params = 0
#         for reshape_layer in model.reshape_layers:
#             num_params += count_parameters(reshape_layer)

#         num_params += count_parameters(model.fusion_net)
#         logger.info("Fusion Model Params: {}".format(num_params) )
#         genotype = model.genotype()
#     logger.info(str(genotype))
#     best_test_f1 = epoch_f1
#     return best_test_f1


