import os, sys
import yaml
import numpy as np

## pymoo: (비지배 정렬 기반)다목적 최적화 알고리즘의 지원!!
from pymoo.factory import get_performance_indicator
from pymoo.util.misc import find_duplicates
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

# from pymoo.configuration import Configuration
# Configuration.show_compile_hint = False

# 경고 메시지 억제 (선택 사항)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Pareto Front(비지배 솔루션의 집합) 찾기 - 최적의 알고리즘 찾아라!!
def find_pareto_front(Y, return_index=False):
    '''
    Find pareto front (undominated part) of the input performance data.
    '''
    if len(Y) == 0: return np.array([])
    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        if not (np.logical_and((Y <= Y[idx]).all(axis=1), (Y < Y[idx]).any(axis=1))).any():
            pareto_indices.append(idx)
    pareto_front = Y[pareto_indices].copy()

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front

# Pareto Frone와 참조점 간 Hypervolume 계산!! -> 최적화 평가 용도
def calc_hypervolume(pfront, ref_point):
    '''
    Calculate hypervolume of pfront based on ref_point
    '''
    hv = get_performance_indicator('hv', ref_point=ref_point)
    return hv.calc(pfront)

# 결과 저장을 위한 디렉터리 경로를 생성 및 반환
def get_result_dir(args):
    '''
    Get directory of result location (result/problem/subfolder/algo/seed/)
    '''
    top_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)), 'result')
    exp_name = '' if args.exp_name is None else '-' + args.exp_name
    algo_name = args.algo + exp_name
    result_dir = os.path.join(top_dir, args.problem, args.subfolder, algo_name, str(args.seed))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

# 설정을 YAML 파일로 저장 ㄱㄱ
def save_args(general_args, framework_args):
    '''
    Save arguments to yaml file
    '''
    all_args = {'general': vars(general_args)}
    all_args.update(framework_args)

    result_dir = get_result_dir(general_args)
    args_path = os.path.join(result_dir, 'args.yml')

    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, 'w') as f:
        yaml.dump(all_args, f, default_flow_style=False, sort_keys=False)

## 파일에 로그를 기록할 수 있는 로거를 설정한다!!
def setup_logger(args):
    '''
    Log to file if needed
    '''
    logger = None

    if args.log_to_file:
        result_dir = get_result_dir(args)
        log_path = os.path.join(result_dir, 'log.txt')
        logger = open(log_path, 'w')
        sys.stdout = logger
    
    return logger

# ---------------------------------------------------------------------------------------------------------
# Survival Selection(진화 알고리즘 관련 함수)
# ---------------------------------------------------------------------------------------------------------

def RankAndCrowdingSurvival_Outer(pop, normalize, n_survive, D=None):

    # get the objective space values and objects
    F = []
    for cfg in pop:
        F.append([cfg['F1-W@1'], cfg['latency'], cfg['energy']])

    F = np.array(F).astype(np.float64, copy=False)
    
    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front
        crowding_of_front = calc_crowding_distance(F[front, :])

        # save rank and crowding in the individual class
        for j, i in enumerate(front):
            pop[i]["rank"] = k
            pop[i]["crowding"] = crowding_of_front[j]

        # current front sorted by crowding distance if splitting
        if len(survivors) + len(front) > n_survive:
            I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
            I = I[:(n_survive - len(survivors))]

        # otherwise take the whole front unsorted
        else:
            I = np.arange(len(front))

        # extend the survivors by all or selected individuals
        survivors.extend(front[I])

    new_pop = []
    for i in survivors:
        new_pop.append(pop[i])
    
    return new_pop

## 성능 우수한 Fusion Network의 선택 함수
def RankAndCrowdingSurvival_Outer_Acc(pop, normalize, n_survive, D=None, perfo_metric='Acc@1'):

    # get the objective space values and objects
    F = []
    for cfg in pop:
        F.append([100-cfg[perfo_metric], cfg['latency'], cfg['energy']])        # 각 개체의 성능 지표(100-acc, lat, ener) 저장
    F = np.array(F).astype(np.float64, copy=False)
    
    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front(비열등 해집합)
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front(개체간 거리 계산=> 다양성 유지 ㄱㄱ)
        crowding_of_front = calc_crowding_distance(F[front, :])

        # save rank and crowding in the individual class
        for j, i in enumerate(front):
            pop[i]["rank"] = k
            pop[i]["crowding"] = crowding_of_front[j]

        # current front sorted by crowding distance if splitting (Crowding Dist가 큰 개체부터 선택 ㄱㄱ)
        if len(survivors) + len(front) > n_survive:
            I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
            I = I[:(n_survive - len(survivors))]

        # otherwise take the whole front unsorted
        else:
            I = np.arange(len(front))

        # extend the survivors by all or selected individuals
        survivors.extend(front[I])

    new_pop = []
    for i in survivors:
        new_pop.append(pop[i])      # 우수한 개체 집합 반환
    
    return new_pop

## 개별 백본 네트워크 선별용 함
def RankAndCrowdingSurvival_Inner_Acc(pop, normalize, n_survive, D=None, perfo_metric='Acc@1'):

    # get the objective space values and objects
    F = []
    for cfg in pop:
        F.append([100-cfg[perfo_metric], cfg['latency'], cfg['energy']])
        
    F = np.array(F).astype(np.float64, copy=False)      # 개별 백본의 성능 지표 저장 (100-acc, lat, ener)
    
    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front (백본들 간 Pareto 군집 생성)
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front (Crowding Dist. 가 큰 개체부터 선택 ㄱㄱ)
        crowding_of_front = calc_crowding_distance(F[front, :])

        # save rank and crowding in the individual class
        for j, i in enumerate(front):
            pop[i]["rank"] = k
            pop[i]["crowding"] = crowding_of_front[j]

        # current front sorted by crowding distance if splitting
        if len(survivors) + len(front) > n_survive:
            I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
            I = I[:(n_survive - len(survivors))]

        # otherwise take the whole front unsorted
        else:
            I = np.arange(len(front))

        # extend the survivors by all or selected individuals
        survivors.extend(front[I])

    new_pop = []
    for i in survivors:
        new_pop.append(pop[i])      # 상위 n_survive 백본 반환!!
    
    return new_pop

def RankAndCrowdingSurvival_Inner(pop, normalize, n_survive, D=None):

    # get the objective space values and objects
    F = []
    if normalize != None:
        for cfg in pop:
            F.append([cfg['latency']/normalize['latency'], cfg['energy']/normalize['energy']])
    else:
        F.append([cfg['latency'], cfg['energy']])

    F = np.array(F).astype(np.float64, copy=False)
    
    # the final indices of surviving individuals
    survivors = []

    # do the non-dominated sorting until splitting front
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

    for k, front in enumerate(fronts):

        # calculate the crowding distance of the front
        crowding_of_front = calc_crowding_distance(F[front, :])

        # save rank and crowding in the individual class
        for j, i in enumerate(front):
            pop[i]["rank"] = k
            pop[i]["crowding"] = crowding_of_front[j]

        # current front sorted by crowding distance if splitting
        if len(survivors) + len(front) > n_survive:
            I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
            I = I[:(n_survive - len(survivors))]

        # otherwise take the whole front unsorted
        else:
            I = np.arange(len(front))

        # extend the survivors by all or selected individuals
        survivors.extend(front[I])

    new_pop = []
    for i in survivors:
        new_pop.append(pop[i])
    
    return new_pop

def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-24)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding