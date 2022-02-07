import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys, os
from collections import namedtuple
import pickle
import math
import numpy as np
import io
try:
    from torch2trt import torch2trt
    HAS_TensorRT = True
except ModuleNotFoundError:
    HAS_TensorRT = False

global SingleData, device, NETWORK_SHAPE
HIGHER_VERSION_PYTORCH = False
SingleData, device, NETWORK_SHAPE = None,None,None
if torch.__version__.startswith('1.0'):
    filter_data_type = torch.uint8
elif torch.__version__.startswith("1.4") or torch.__version__.startswith("1.5") or torch.__version__.startswith("1.6"):
    filter_data_type = torch.bool
    HIGHER_VERSION_PYTORCH = True
    print("Using higher pytorch version {}".format(torch.__version__))
else:
    filter_data_type = torch.bool

def initialQLearning(force_cpu=False, designated_gpu:int = None) -> str:
    global SingleData, device, NETWORK_SHAPE
    SingleData = namedtuple('SingleData',
                            ('current_state', 'action', 'new_state', 'reward'))
    if not force_cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    if torch.cuda.is_available() and not force_cpu:
        if torch.cuda.device_count() > 1:
            if designated_gpu is None:
                selectGPU()
            else:
                selectGPU(force=designated_gpu)
        print("    Model learning with GPU: " + torch.cuda.get_device_name(device))
        return torch.cuda.get_device_name(device)
    else:
        print("    Model learning with CPU")
        import cpuinfo
        try:
            return cpuinfo.get_cpu_info()['brand']
        except KeyError:
            return cpuinfo.get_cpu_info()['brand_raw']
        # return None
    # device = torch.device("cpu")   #Temperary Test
    # NETWORK_SHAPE = [(5,20),(20,40),(40,80),(80,160),(160,320),(320,320),(320,640),(640,11**3)]

class QNetworkModel(nn.Module):
    def __init__(self):
        super(QNetworkModel, self).__init__()
        self.func_in = nn.Linear(*NETWORK_SHAPE[0])
        self.hiddn_layers = []
        for layer in range(len(NETWORK_SHAPE) - 2 ):
            self.add_module('h_layer_' + str(layer), nn.Linear(*NETWORK_SHAPE[layer+1]))
            self.hiddn_layers.append(getattr(self, 'h_layer_' + str(layer)))
        self.func_out = nn.Linear(*NETWORK_SHAPE[-1])

    def forward(self, x):
        x = F.relu(self.func_in(x))
        for layer in self.hiddn_layers:
            x = F.relu(layer(x))
        x = self.func_out(x)
        return x

class QNetworkModelDP(nn.Module):
    def __init__(self):
        super(QNetworkModelDP, self).__init__()
        self.func_in = nn.Linear(*NETWORK_SHAPE[0])
        self.hiddn_layers = []
        for layer in range(len(NETWORK_SHAPE) - 2 ):
            self.add_module('h_layer_' + str(layer), nn.Linear(*NETWORK_SHAPE[layer+1]))
            self.hiddn_layers.append(getattr(self, 'h_layer_' + str(layer)))
        self.func_out = nn.Linear(*NETWORK_SHAPE[-1])
        self.dropout_rate = 0.3

    def forward(self, x):
        x = F.relu(self.func_in(x))
        for layer in self.hiddn_layers:
            x = F.relu(layer(x))
            x = F.dropout(x, self.dropout_rate)
        x = self.func_out(x)
        return x



class QNetworkModelManual(nn.Module):
    def __init__(self):
        super(QNetworkModelManual, self).__init__()
        self.func_in = nn.Linear(*NETWORK_SHAPE[0])
        self.fcn1 = nn.Linear(*NETWORK_SHAPE[1])
        self.fcn2 = nn.Linear(*NETWORK_SHAPE[2])
        self.fcn3 = nn.Linear(*NETWORK_SHAPE[3])
        self.fcn4 = nn.Linear(*NETWORK_SHAPE[4])
        self.fcn5 = nn.Linear(*NETWORK_SHAPE[5])
        self.fcn6 = nn.Linear(*NETWORK_SHAPE[6])
        self.func_out = nn.Linear(*NETWORK_SHAPE[-1])

    
    def forward(self, x):
        x = F.relu(self.func_in(x))
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = F.relu(self.fcn3(x))
        x = F.relu(self.fcn4(x))
        x = F.relu(self.fcn5(x))
        x = F.relu(self.fcn6(x))
        x = self.func_out(x)
        return x

class LearningMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, currentState, action, newState, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        if not HIGHER_VERSION_PYTORCH:
            tensor_transition = SingleData(
                torch.tensor([currentState], device=device, dtype=torch.float),
                torch.tensor([[action]], device=device, dtype=torch.long),
                torch.tensor([newState], device=device, dtype=torch.float) if newState is not None else None,
                torch.tensor([reward], device=device, dtype=torch.float)
            )
        else:
            tensor_transition = SingleData(
                torch.tensor([currentState], device=device),
                torch.tensor([[action]], device=device, dtype=torch.long),
                torch.tensor([newState], device=device) if newState is not None else None,
                torch.tensor([reward], device=device, dtype=torch.double)
            )
        self.memory[self.position] = tensor_transition
        self.position = (self.position + 1) % self.capacity 

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

global policy_QNetwork, target_QNetwork, optimizer
policy_QNetwork, target_QNetwork, optimizer = None, None, None
tensorRT_QNetwork = None

def InitialQNetwork(shape, designative_mode = False, model_name=None, dropout_rate=None):
    global policy_QNetwork, target_QNetwork, NETWORK_SHAPE
    result = False
    dropout_activated = False
    generateNetworkShape(shape)
    
    if (dropout_rate is not None and isinstance(dropout_rate,float)
        and dropout_rate < 1 and dropout_rate >= 0
    ):
        PolicyModel = globals()['QNetworkModelDP']
        dropout_activated = True
        print("    Activate Dropout Model")
    else:
        if dropout_rate is not None:
            print("    Dropout Rate Error")
        PolicyModel = globals()['QNetworkModel']

    if not designative_mode:
        for name in os.listdir("./model"):
            if name.endswith('.model'): print(name[:-6])
        filename = input('input the .parm file or leave blank to creat new Q_Network:')
    else:
        filename = model_name
    
    try:
        shape_filename = "./model/" + filename + ".shape"
        with open(shape_filename,'rb') as f:
            read_NETWORK_SHAPE = pickle.load(f)
        if read_NETWORK_SHAPE != NETWORK_SHAPE:
            print("Network shape miss match, please check learning setting.")
            print("Shape of %s:" %(filename))
            print("    "+str(read_NETWORK_SHAPE))
            print("Learning Setting:")
            print("    "+str(NETWORK_SHAPE))
            if not designative_mode:
                sys.exit(0)
            else:
                return result
        else:
            model_filename = "./model/" + filename + ".model"
            # policy_QNetwork = QNetworkModelManual()
            # policy_QNetwork = QNetworkModel()
            policy_QNetwork = PolicyModel()
            if torch.cuda.is_available():
                try:
                    policy_QNetwork.load_state_dict(torch.load(model_filename))
                except:
                    policy_QNetwork.load_state_dict(torch.load(model_filename, map_location=device))
            else:
                policy_QNetwork.load_state_dict(torch.load(model_filename, map_location=device))
            print('    Read Q_Parm successfulluy...')


    except FileNotFoundError: 
        #initial QTable
        # policy_QNetwork = QNetworkModelManual()
        # policy_QNetwork = QNetworkModel()
        policy_QNetwork = PolicyModel()
        print('    Creating... new Q_Network...')

    if dropout_activated:
        policy_QNetwork.dropout_rate = dropout_rate
        print("    Set Dropout Rate to :{0:.2f}".format(dropout_rate))

    # target_QNetwork = QNetworkModelManual()
    # target_QNetwork = QNetworkModel()
    target_QNetwork = QNetworkModel()
    target_QNetwork.load_state_dict(policy_QNetwork.state_dict())
    policy_QNetwork.to(device)
    target_QNetwork.to(device)
    target_QNetwork.eval()
    global optimizer
    optimizer = torch.optim.Adam(policy_QNetwork.parameters(), lr=learning_rate)
    result = True
    return result

def InitialQNetworkInExcuMode(shape):
    global policy_QNetwork
    generateNetworkShape(shape)
    policy_QNetwork = QNetworkModel()
    policy_QNetwork.to(device)
    policy_QNetwork.eval()

def changeModeltoExcuMode(half_mode=False):
    if half_mode: policy_QNetwork.half()
    policy_QNetwork.eval()
    policy_QNetwork.to(device)

def loadStateDict(state_direct, trt_mode=False, half_mode=False):
    policy_QNetwork.load_state_dict(state_direct)
    if half_mode: policy_QNetwork.half()
    policy_QNetwork.eval()
    policy_QNetwork.to(device)
    if trt_mode: loadModeltoTensorRT(half_mode)
    # policy_QNetwork.load_state_dict(torch.load(pickle.dump(state_direct), map_location=device))
    # policy_QNetwork.load_state_dict(torch.load(state_direct, map_location=device))

def loadModeltoTensorRT(half_mode=False):
    global tensorRT_QNetwork
    if HAS_TensorRT:
        dummy_state = torch.tensor(np.zeros((1,NETWORK_SHAPE[0][0])), device=device).float()
        if half_mode: dummy_state = dummy_state.half().to(device)
        try:
            tensorRT_QNetwork = torch2trt(policy_QNetwork, [dummy_state])
        except RuntimeError as err:
            print(err)
            print(dummy_state.type())
            sys.exit()

        print("Modle load as TensorRT mode")
    else:
        print('TensorRT has encounted error')
        pass

def getStateDictBuffer():
    if policy_QNetwork is not None:
        # buffer = io.BytesIO()
        # torch.save(policy_QNetwork.state_dict(), buffer)
        # return buffer
        return policy_QNetwork.state_dict()

###===============DQN Settings===============###

learning_rate = 1e-4
BATCH_SIZE = 256
GAMMA = 0.7
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500

###===============DQN Settings===============###

def Learning(memory):
    if len(memory) < BATCH_SIZE:
        return None
    else:
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = SingleData(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.new_state)), device=device, dtype=filter_data_type)
        non_final_next_states = torch.cat([s for s in batch.new_state
                                                    if s is not None])
        state_batch  = torch.cat(batch.current_state)
        action_batch = torch.cat(batch.action)

        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_QNetwork(state_batch).gather(1, action_batch)
        if HIGHER_VERSION_PYTORCH:state_action_values = state_action_values.double()
        

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_QNetwork(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss_return = loss.item()
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # resticct between -1~1
        for param in policy_QNetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss_return

def updateTarget():
    target_QNetwork.load_state_dict(policy_QNetwork.state_dict())

def choseAction(state, n_action, target_done=None, in_learning=True, half_mode=False):
    state = torch.tensor(state, device = device) if not half_mode else (
        torch.tensor(state, device = device).half()
    )
    if in_learning:
        sample = random.random()
        #greedy, when under learning, target_done shouldn't be None
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * target_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # return policy_QNetwork(state).max(1)[1].view(1, 1).item()
                return torch.argmax(policy_QNetwork(state)).item(), False, eps_threshold
        else:
            return random.randrange(n_action), True, eps_threshold
    else:
        #Off Learning Mode
        with torch.no_grad():
            return torch.argmax(policy_QNetwork(state)).item(), False, 0
    
    # action_number, random_flag, eps_threshold

def getQRaw(state, half_mode=False):
    state = torch.tensor(state, device = device) if not half_mode else (
        torch.tensor(state, device = device).half()
    )
    with torch.no_grad():
        return policy_QNetwork(state).cpu().numpy()

def choseActionTRT(state, n_action, target_done=None, in_learning=True, half_mode=False):
    state = torch.tensor(np.array(state).reshape(1,len(state)), device = device).float()
    if half_mode: state = state.half().to(device)
    if in_learning:
        sample = random.random()
        #greedy, when under learning, target_done shouldn't be None
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * target_done / EPS_DECAY)
        if sample > eps_threshold:
            if half_mode:
                return np.argmax(tensorRT_QNetwork(state).cpu().numpy()), False, eps_threshold
            else:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    # return policy_QNetwork(state).max(1)[1].view(1, 1).item()
                    return torch.argmax(tensorRT_QNetwork(state)).item(), False, eps_threshold
        else:
            return random.randrange(n_action), True, eps_threshold
    else:
        #Off Learning Mode
        if half_mode:
            return np.argmax(tensorRT_QNetwork(state).cpu().numpy()), False, 0
        else:
            with torch.no_grad():
                return torch.argmax(tensorRT_QNetwork(state)).item(), False, 0
    
    # action_number, random_flag, eps_threshold

def getQRawTRT(state, half_mode=False):
    state = torch.tensor(np.array(state).reshape(1,len(state)), device = device).float()
    with torch.no_grad():
        return tensorRT_QNetwork(state).cpu().numpy()

def modelWarmUp(warm_up_times, state_number, half_mode=False):
    states = np.random.rand(warm_up_times, state_number).astype('float32')
    states = states*50 #into our definition space
    for index in range(warm_up_times):
        choseAction(
            states[index],
            None,
            in_learning=False,
            half_mode=half_mode
        )

def modelWarmUpTRT(warm_up_times, state_number, half_mode=False):
    states = np.random.rand(warm_up_times, state_number).astype('float32')
    states = states*50 #into our definition space
    for index in range(warm_up_times):
        choseActionTRT(
            states[index],
            None,
            in_learning=False,
            half_mode=half_mode
        )

def SaveModel(fileName, conf = None, remove_temp = False):
    if 'policy_QNetwork' in globals():
        if conf is not None:
            conf.name = fileName
        fileName = "./model/" + fileName
        torch.save(policy_QNetwork.state_dict(), fileName + ".model")
        with open(fileName + ".shape", 'wb+') as f:
            pickle.dump(NETWORK_SHAPE,f)
        if conf is not None:
            with open(fileName + ".lrconf", 'wb+') as f:
                pickle.dump(conf,f)
        if remove_temp:
            try:
                os.remove("./model/temp/temp_{0}.model".format(os.getpid()))
                os.remove("./model/temp/temp_{0}.lrconf".format(os.getpid()))
                os.remove("./model/temp/temp_{0}.shape".format(os.getpid()))
                print("temp file removed")
            except FileNotFoundError:
                print("No temp file to be removed")
        
def policyEvalOnly():
    policy_QNetwork.eval()
    print("Policy Model set to Eval mode")

def chengeBatchSize(size):
    global BATCH_SIZE
    BATCH_SIZE = size

def changeEpsDecay(decay_step):
    global EPS_DECAY
    EPS_DECAY = decay_step

def changeEpsStart(eps_start):
    """(1.0~0.0) In float form"""
    global EPS_START
    EPS_START = eps_start

def changeEpsEnd(eps_end: float):
    """(1.0~0.0) In float form"""
    global EPS_END
    EPS_END = eps_end

def randAction(State,n_action):
    return random.randrange(n_action)

def generateNetworkShape(shape):
    global NETWORK_SHAPE
    NETWORK_SHAPE = list()
    for index, value in enumerate(shape):
        if index != len(shape)-1:
            NETWORK_SHAPE.append((value,shape[index+1]))


def testGround():
    # InitialQNetwork(5,1331)
    # print(policy_QNetwork.hiddn_layers[0].state_dict())
    # SaveModel("savetest")
    # InitialQNetwork(5,1331)
    # print(policy_QNetwork.h_layer_0.state_dict())
    shape = [
            5,
            20,40,80,160,320,320,640,
            31**3
        ]
    generateNetworkShape(shape)
    print(shape)
    print(NETWORK_SHAPE)

def selectGPU(force=None):
    if force is None:
        gpu_candidate = list()
        for gpu_index in range(torch.cuda.device_count()):
            print("    device [{0}]: {1}".format(gpu_index, torch.cuda.get_device_name(gpu_index)))
            gpu_candidate.append(gpu_index)
        select = -1
        while select not in gpu_candidate:
            select = int(input("Please select your device:"))
        torch.cuda.set_device(select)
    else:
        torch.cuda.set_device(force)

def debug(var):
    print(var)
    sys.exit()

class SizeMissMatch(Exception):
    """Exception for wrong input"""

if __name__ == "__main__":
    # testGround()
    import LearningUtility as LUtil
    from LearningUtility import TrainingConfiguration

    model_name = 'cycloid_gait_11'
    config = LUtil.loadOldConfig(model_name)
    initialQLearning()

    InitialQNetwork(config.network_shape, designative_mode=True, model_name=model_name)
    policyEvalOnly()

    loadModeltoTensorRT()


